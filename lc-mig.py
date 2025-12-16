#!/usr/bin/env python3

import argparse
import hashlib
import json
import os
import secrets
import shutil
import sys
import uuid
from collections import Counter, defaultdict
from collections.abc import Iterable, Iterator
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import requests
from bson import ObjectId
from pymongo import ASCENDING, MongoClient

try:
    import psycopg
    from psycopg.rows import dict_row
except ImportError:  # pragma: no cover - optional dependency
    psycopg = None
    dict_row = None


def load_env(path: str = ".env"):
    if not os.path.exists(path):
        return
    with open(path) as f:
        for raw in f:
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if not key or key in os.environ:
                continue
            cleaned = value.strip().strip('"').strip("'")
            os.environ[key] = cleaned


def get_session_headers(content_type: str | None = "application/json"):
    api_key = os.environ.get("ONYX_API_KEY")
    if not api_key:
        raise RuntimeError("ONYX_API_KEY missing")
    headers = {"Authorization": f"Bearer {api_key}"}
    if content_type:
        headers["Content-Type"] = content_type
    return headers


def fetch_existing_users(api_base: str):
    headers = get_session_headers()
    resp = requests.get(
        f"{api_base}/api/manage/users",
        headers=headers,
        params={"include_api_keys": "false"},
    )
    resp.raise_for_status()
    data = resp.json()
    emails = set()
    for key in ("accepted", "invited", "slack_users"):
        for entry in data.get(key, []):
            email = (entry.get("email") or "").lower()
            if email:
                emails.add(email)
    return emails


def fetch_librechat_users(mongo_uri: str):
    client = MongoClient(mongo_uri, connect=False)
    db = client.get_database("LibreChat")
    for doc in db.users.find({}):
        email = (doc.get("email") or "").strip()
        if not email:
            continue
        yield {
            "email": email,
            "name": doc.get("name") or "",
            "role": doc.get("role") or "BASIC",
        }


def register_user(api_base: str, email: str, password: str):
    headers = get_session_headers()
    payload = {
        "email": email,
        "password": password,
        "is_active": True,
        "is_superuser": False,
        "is_verified": True,
        "role": "basic",
    }
    resp = requests.post(f"{api_base}/api/auth/register", headers=headers, json=payload)
    if resp.status_code == 400 and "REGISTER_USER_ALREADY_EXISTS" in resp.text:
        return None
    resp.raise_for_status()
    return resp.json()


def log_credentials(out_path, record):
    with open(out_path, "a") as f:
        f.write(json.dumps(record))
        f.write("\n")


def create_users(args):
    api_base = os.environ.get("ONYX_API_BASE")
    mongo_uri = os.environ.get("LIBRECHAT_MONGODB")
    if not api_base or not mongo_uri:
        raise RuntimeError("ONYX_API_BASE and LIBRECHAT_MONGODB must be set")

    existing = fetch_existing_users(api_base)
    skip_emails = {email.lower() for email in args.skip_email or []}

    created_log = (
        args.output
        or f"created_users_{datetime.utcnow().strftime('%Y%m%d%H%M%S')}.jsonl"
    )
    os.makedirs(os.path.dirname(created_log) or ".", exist_ok=True)

    total_created = 0
    for user in fetch_librechat_users(mongo_uri):
        email_lc = user["email"].lower()
        if email_lc in existing or email_lc in skip_emails:
            continue
        temp_password = f"{secrets.token_urlsafe(12)}!"
        try:
            register_user(api_base, user["email"], temp_password)
            log_credentials(
                created_log,
                {
                    "email": user["email"],
                    "password": temp_password,
                    "role": "basic",
                },
            )
            total_created += 1
            print(f"Created user {user['email']}")
        except requests.HTTPError as http_err:
            response_text = ""
            if http_err.response is not None:
                response_text = getattr(http_err.response, "text", "")
            print(
                f"Failed to create {user['email']}: {http_err} {response_text}",
                file=sys.stderr,
            )
    print(f"Done. Created {total_created} users. Credentials logged to {created_log}")


def _slugify_email(email: str) -> str:
    safe = []
    for ch in email.lower():
        if ch.isalnum():
            safe.append(ch)
        elif ch in {"@", ".", "-", "_"}:
            safe.append("-")
    slug = "".join(safe).strip("-")
    return slug or "user"


def _normalize_value(value: Any):
    if isinstance(value, ObjectId):
        return str(value)
    if isinstance(value, datetime):
        if value.tzinfo is None:
            value = value.replace(tzinfo=UTC)
        return value.astimezone(UTC).isoformat()
    if isinstance(value, list):
        return [_normalize_value(v) for v in value]
    if isinstance(value, dict):
        return {k: _normalize_value(v) for k, v in value.items()}
    return value


def _normalize_doc(doc: dict[str, Any]) -> dict[str, Any]:
    return {key: _normalize_value(value) for key, value in doc.items()}


def _extract_message_file_ids(message: dict[str, Any]) -> list[str]:
    ids: list[str] = []

    for entry in message.get("files") or []:
        fid = entry.get("file_id") or entry.get("fileId")
        if fid:
            ids.append(fid)

    for entry in message.get("attachments") or []:
        fid = entry.get("file_id") or entry.get("fileId")
        if fid:
            ids.append(fid)

    return ids


def _resolve_file_path(
    file_doc: dict[str, Any], candidate_roots: list[Path]
) -> Path | None:
    raw_path = file_doc.get("filepath") or file_doc.get("path")
    filename = file_doc.get("filename")
    user_id = file_doc.get("user")

    possible_paths: list[Path] = []
    if raw_path:
        raw_path = str(raw_path)
        path_obj = Path(raw_path)
        if path_obj.is_file():
            possible_paths.append(path_obj)
        if raw_path.startswith("/"):
            trimmed = raw_path.lstrip("/")
        else:
            trimmed = raw_path
        for root in candidate_roots:
            possible_paths.append(root / trimmed)

    if filename:
        for root in candidate_roots:
            if user_id:
                possible_paths.append(root / str(user_id) / filename)
            possible_paths.append(root / filename)

    for candidate in possible_paths:
        if candidate.is_file():
            return candidate

    return None


def _sha256_of_file(path: Path) -> str | None:
    if not path or not path.is_file():
        return None
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _analyze_branching(messages: list[dict[str, Any]]) -> dict[str, Any]:
    parent_counts: Counter[str] = Counter()
    seen_ids = set()
    missing_parent_ids = set()
    for msg in messages:
        msg_id = msg.get("messageId")
        if msg_id:
            seen_ids.add(msg_id)
        parent_id = msg.get("parentMessageId")
        if parent_id:
            parent_counts[parent_id] += 1

    for parent_id, _count in parent_counts.items():
        if parent_id and parent_id not in seen_ids:
            missing_parent_ids.add(parent_id)

    branching_points = [
        parent_id
        for parent_id, count in parent_counts.items()
        if parent_id and count > 1
    ]

    return {
        "has_branching": bool(branching_points),
        "branch_points": branching_points,
        "missing_parent_ids": sorted(missing_parent_ids),
    }


def _split_messages_into_branches(messages: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    """Return a list of linear message sequences, one per branch."""
    id_to_msg: dict[str, dict[str, Any]] = {}
    order_index: dict[str, int] = {}
    for idx, message in enumerate(messages):
        msg_id = message.get("messageId")
        if not msg_id:
            continue
        id_to_msg[msg_id] = message
        order_index[msg_id] = idx

    if not id_to_msg:
        return []

    children: dict[str, list[str]] = defaultdict(list)
    roots: list[str] = []
    for message in messages:
        msg_id = message.get("messageId")
        if not msg_id:
            continue
        parent_id = message.get("parentMessageId")
        if parent_id and parent_id in id_to_msg:
            children[parent_id].append(msg_id)
        else:
            roots.append(msg_id)

    if not roots:
        # Fallback: treat the earliest message as the root
        roots = (
            [messages[0]["messageId"]]
            if messages and messages[0].get("messageId")
            else list(id_to_msg.keys())
        )

    def dfs(current_id: str, current_path: list[str]) -> list[list[str]]:
        new_path = current_path + [current_id]
        child_ids = children.get(current_id)
        if not child_ids:
            return [new_path]
        ordered_children = sorted(child_ids, key=lambda cid: order_index.get(cid, 0))
        paths: list[list[str]] = []
        for child_id in ordered_children:
            paths.extend(dfs(child_id, new_path))
        return paths

    ordered_roots = sorted(set(roots), key=lambda rid: order_index.get(rid, 0))
    branch_id_paths: list[list[str]] = []
    seen_paths: set[tuple[str, ...]] = set()
    for root_id in ordered_roots:
        for path in dfs(root_id, []):
            path_key = tuple(path)
            if path_key not in seen_paths:
                seen_paths.add(path_key)
                branch_id_paths.append(path)

    if not branch_id_paths:
        return []

    variants: list[list[dict[str, Any]]] = []
    for path in branch_id_paths:
        path_ids = set(path)
        variant_messages: list[dict[str, Any]] = []
        for message in messages:
            msg_id = message.get("messageId")
            if not msg_id or msg_id in path_ids:
                variant_messages.append(message)
        variants.append(variant_messages)
    return variants


def _gather_asset_roots(args) -> list[Path]:
    roots: list[str] = []
    env_roots = os.environ.get("LIBRECHAT_ASSET_ROOTS")
    if args.asset_root:
        roots.extend(args.asset_root)
    elif env_roots:
        roots.extend(env_roots.split(os.pathsep))
    default_roots = [
        "../LibreChat/uploads",
        "../LibreChat/images",
        "../LibreChat",
    ]
    if not roots:
        roots.extend(default_roots)

    resolved = []
    for raw in roots:
        path = Path(raw).expanduser().resolve()
        if path.exists():
            resolved.append(path)
    return resolved or [Path(r).expanduser().resolve() for r in default_roots]


def export_chats(args):
    mongo_uri = os.environ.get("LIBRECHAT_MONGODB")
    if not mongo_uri:
        raise RuntimeError("LIBRECHAT_MONGODB must be set")

    if not args.user_email:
        raise RuntimeError("At least one --user-email must be provided")

    client = MongoClient(mongo_uri, connect=False)
    db = client.get_database("LibreChat")

    lowered_targets = {
        email.strip().lower() for email in args.user_email if email and email.strip()
    }
    if not lowered_targets:
        raise RuntimeError("No valid --user-email values provided")
    user_query = {
        "$expr": {
            "$in": [
                {"$toLower": {"$ifNull": ["$email", ""]}},
                list(lowered_targets),
            ]
        }
    }
    user_docs = list(db.users.find(user_query))

    if not user_docs:
        raise RuntimeError("No LibreChat users matched the provided emails")

    export_root = Path(args.output_dir).expanduser()
    timestamp = datetime.utcnow().strftime("%Y%m%d%H%M%S")
    export_root = export_root / f"export_{timestamp}"
    export_root.mkdir(parents=True, exist_ok=True)

    asset_roots = _gather_asset_roots(args)

    for user in user_docs:
        email = user.get("email") or "unknown"
        user_slug = _slugify_email(email)
        user_dir = export_root / user_slug
        user_dir.mkdir(parents=True, exist_ok=True)

        user_summary = {
            "exported_at": datetime.utcnow().isoformat(),
            "email": email,
            "user_id": str(user.get("_id")),
            "role": user.get("role"),
            "provider": user.get("provider"),
            "conversation_count": 0,
            "message_count": 0,
            "file_count": 0,
            "asset_roots_checked": [str(root) for root in asset_roots],
        }

        conversations_path = user_dir / "conversations.jsonl"
        files_manifest_path = user_dir / "files_manifest.jsonl"
        files_dir = user_dir / "files"
        files_dir.mkdir(parents=True, exist_ok=True)

        all_files: dict[str, dict[str, Any]] = {}

        with conversations_path.open("w", encoding="utf-8") as convo_out:
            user_id_str = str(user.get("_id"))
            convo_query = {"user": user_id_str}
            cursor = db.conversations.find(convo_query).sort("createdAt", ASCENDING)
            if args.max_conversations and args.max_conversations > 0:
                cursor = cursor.limit(args.max_conversations)

            for conversation in cursor:
                conv_id = conversation.get("conversationId")
                if not conv_id:
                    continue

                messages = list(
                    db.messages.find({"conversationId": conv_id}).sort(
                        "createdAt", ASCENDING
                    )
                )

                branch_info = _analyze_branching(messages)

                safe_conversation = _normalize_doc(conversation)
                safe_messages = [_normalize_doc(msg) for msg in messages]

                file_ids = sorted(
                    {
                        fid
                        for msg in messages
                        for fid in _extract_message_file_ids(msg)
                        if fid
                    }
                )
                missing_file_ids: list[str] = []
                file_payloads: list[str] = []

                if file_ids:
                    for file_doc in db.files.find({"file_id": {"$in": file_ids}}):
                        file_id = file_doc.get("file_id")
                        if not file_id:
                            continue
                        if file_id in all_files:
                            file_payloads.append(file_id)
                            continue

                        resolved_path = _resolve_file_path(file_doc, asset_roots)
                        sha256 = (
                            _sha256_of_file(resolved_path) if resolved_path else None
                        )
                        errors: list[str] = []
                        copied_rel_path = None

                        if not resolved_path:
                            errors.append("source_file_not_found")
                        elif args.copy_files:
                            target_subdir = files_dir / file_id
                            target_subdir.mkdir(parents=True, exist_ok=True)
                            target_path = target_subdir / (
                                file_doc.get("filename") or file_id
                            )
                            try:
                                shutil.copy2(resolved_path, target_path)
                                copied_rel_path = str(target_path.relative_to(user_dir))
                            except OSError as err:
                                errors.append(f"copy_failed:{err}")

                        payload = {
                            "file_id": file_id,
                            "metadata": _normalize_doc(file_doc),
                            "source_path": str(resolved_path)
                            if resolved_path
                            else None,
                            "sha256": sha256,
                            "copied_rel_path": copied_rel_path,
                            "copied": bool(copied_rel_path),
                            "errors": errors,
                        }
                        all_files[file_id] = payload
                        file_payloads.append(file_id)

                    missing_file_ids = sorted(set(file_ids) - set(file_payloads))

                convo_record = {
                    "exported_at": datetime.utcnow().isoformat(),
                    "user_email": email,
                    "conversation": safe_conversation,
                    "messages": safe_messages,
                    "message_count": len(messages),
                    "file_ids": file_payloads,
                    "missing_file_ids": missing_file_ids,
                    "branching": branch_info,
                }

                convo_out.write(json.dumps(convo_record))
                convo_out.write("\n")

                user_summary["conversation_count"] += 1
                user_summary["message_count"] += len(messages)

        with files_manifest_path.open("w", encoding="utf-8") as files_out:
            for payload in all_files.values():
                files_out.write(json.dumps(payload))
                files_out.write("\n")
        user_summary["file_count"] = len(all_files)

        summary_path = user_dir / "summary.json"
        summary_path.write_text(json.dumps(user_summary, indent=2), encoding="utf-8")

        print(
            f"Exported {user_summary['conversation_count']} conversations / "
            f"{user_summary['message_count']} messages for {email} -> {user_dir}"
        )

    client.close()
    print(f"All exports written to {export_root}")


def _require_psycopg():
    if psycopg is None:
        raise RuntimeError(
            "psycopg is required for import operations. Install with 'pip install psycopg[binary]'."
        )


def _load_jsonl(path: Path) -> Iterator[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _parse_iso_datetime(value: str | None) -> datetime:
    if not value:
        return datetime.utcnow().replace(tzinfo=UTC)
    if isinstance(value, datetime):
        dt = value
    else:
        cleaned = value.replace("Z", "+00:00")
        dt = datetime.fromisoformat(cleaned)
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=UTC)
    return dt.astimezone(UTC)


def _to_naive(dt: datetime | None) -> datetime | None:
    if dt is None:
        return None
    if dt.tzinfo:
        return dt.astimezone(UTC).replace(tzinfo=None)
    return dt


def _safe_uuid(value: str | None) -> uuid.UUID:
    if isinstance(value, uuid.UUID):
        return value
    if value:
        try:
            return uuid.UUID(str(value))
        except (ValueError, TypeError):
            pass
    return uuid.uuid4()


def _coerce_text_value(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, list):
        pieces: list[str] = []
        for item in value:
            piece = _coerce_text_value(item)
            if piece:
                pieces.append(piece)
        if pieces:
            return "\n".join(pieces)
        return None
    if isinstance(value, dict):
        for key in ("value", "text", "content", "message", "body"):
            candidate = _coerce_text_value(value.get(key))
            if candidate:
                return candidate
        if "parts" in value and isinstance(value["parts"], list):
            pieces: list[str] = []
            for part in value["parts"]:
                piece = _coerce_text_value(part)
                if piece:
                    pieces.append(piece)
            if pieces:
                return "\n".join(pieces)
        if "data" in value:
            candidate = _coerce_text_value(value.get("data"))
            if candidate:
                return candidate
        try:
            return json.dumps(value, ensure_ascii=False)
        except TypeError:
            return str(value)
    return None


def _summarize_attachments(
    attachments: Iterable[dict[str, Any]], label: str
) -> str | None:
    rows: list[str] = []
    for attachment in attachments:
        if not isinstance(attachment, dict):
            continue
        name = (
            attachment.get("name")
            or attachment.get("filename")
            or attachment.get("file_id")
            or attachment.get("id")
            or "attachment"
        )
        content_type = (
            attachment.get("type")
            or attachment.get("content_type")
            or attachment.get("mimeType")
        )
        size = attachment.get("bytes") or attachment.get("size")
        extras: list[str] = []
        if content_type:
            extras.append(content_type)
        if size:
            extras.append(f"{size} bytes")
        descriptor = f"{name}"
        if extras:
            descriptor += f" ({', '.join(extras)})"
        rows.append(f"- {descriptor}")
    if not rows:
        return None
    return f"{label}:\n" + "\n".join(rows)


def _message_text_from_record(record: dict[str, Any]) -> str:
    base = record.get("text")
    if isinstance(base, str) and base.strip():
        return base

    content_entries = record.get("content") or []
    parts: list[str] = []
    for entry in content_entries:
        entry_type = entry.get("type")
        if entry_type == "text":
            text_value = _coerce_text_value(entry.get("text"))
            if text_value and text_value.strip():
                parts.append(text_value)
        elif entry_type == "think":
            think = _coerce_text_value(entry.get("think"))
            if think:
                sanitized = think.strip()
                quoted = "\n".join(f"> {line}" if line else ">" for line in sanitized.splitlines())
                parts.append(f"> **Thinking**\n{quoted}")
        elif entry_type == "tool_call":
            tool_call = entry.get("tool_call") or {}
            name = tool_call.get("name") or tool_call.get("type") or "tool_call"
            parts.append(f"[{name}] {json.dumps(tool_call, ensure_ascii=False)}")
        elif entry_type == "attachments":
            summary = _summarize_attachments(entry.get("attachments") or [], "Attachments")
            if summary:
                parts.append(summary)
        elif entry_type == "error":
            error_text = _coerce_text_value(entry.get("error"))
            if error_text:
                parts.append(f"[error] {error_text}")
        elif entry_type == "agent_update":
            agent_update = entry.get("agent_update") or {}
            parts.append(f"[agent_update] {json.dumps(agent_update, ensure_ascii=False)}")
        elif entry_type == "image_file":
            summary = _summarize_attachments(
                [entry.get("image_file") or {}], "Image"
            )
            if summary:
                parts.append(summary)
        else:
            fallback = _coerce_text_value(entry.get(entry_type))
            if fallback:
                parts.append(f"[{entry_type}] {fallback}")

    if parts:
        return "\n\n".join(parts)

    message_id = record.get("messageId") or "unknown"
    finish_reason = record.get("finish_reason")
    if isinstance(finish_reason, str) and finish_reason.strip():
        return f"[finish_reason={finish_reason} for messageId={message_id}]"

    return f"[empty message imported from LibreChat messageId={message_id}]"


def _chat_metadata_payload(conversation: dict[str, Any]) -> dict[str, Any]:
    return {
        "librechat": {
            key: conversation.get(key)
            for key in (
                "conversationId",
                "endpoint",
                "model",
                "assistant_id",
                "agent_id",
                "spec",
                "tags",
                "web_search",
                "files",
                "file_ids",
                "title",
                "greeting",
                "promptPrefix",
                "useResponsesApi",
            )
            if key in conversation
        }
    }


def _message_metadata_payload(message: dict[str, Any]) -> dict[str, Any]:
    return {
        "messageId": message.get("messageId"),
        "sender": message.get("sender"),
        "endpoint": message.get("endpoint"),
        "model": message.get("model"),
        "attachments": message.get("attachments"),
        "content": message.get("content"),
        "parentMessageId": message.get("parentMessageId"),
        "thread_id": message.get("thread_id"),
    }


def _guess_chat_file_type(content_type: str | None) -> str:
    if not content_type:
        return "file"
    lowered = content_type.lower()
    if lowered.startswith("image/"):
        return "image"
    if lowered.startswith("audio/"):
        return "audio"
    if lowered.startswith("video/"):
        return "video"
    if lowered in {"application/pdf", "text/plain", "text/markdown"}:
        return "document"
    return "file"


def _ensure_metadata_table(cursor):
    cursor.execute(
        """
        CREATE TABLE IF NOT EXISTS librechat_message_metadata (
            chat_message_id integer PRIMARY KEY
                REFERENCES chat_message(id) ON DELETE CASCADE,
            metadata jsonb NOT NULL
        )
        """
    )


def _resolve_file_source(bundle_root: Path, entry: dict[str, Any]) -> Path | None:
    rel_path = entry.get("copied_rel_path")
    if rel_path:
        candidate = bundle_root / rel_path
        if candidate.is_file():
            return candidate
    source = entry.get("source_path")
    if source:
        candidate = Path(source).expanduser()
        if candidate.is_file():
            return candidate
    return None


def import_chats(args):
    load_env(".env")
    bundle_path = Path(args.bundle_path).expanduser()
    if not bundle_path.exists():
        raise RuntimeError(f"Bundle path not found: {bundle_path}")

    summary_path = bundle_path / "summary.json"
    conversations_path = bundle_path / "conversations.jsonl"
    files_manifest_path = bundle_path / "files_manifest.jsonl"

    if not summary_path.is_file():
        raise RuntimeError(f"summary.json missing under {bundle_path}")
    if not conversations_path.is_file():
        raise RuntimeError(f"conversations.jsonl missing under {bundle_path}")

    summary = json.loads(summary_path.read_text(encoding="utf-8"))
    bundle_email = summary.get("email")

    target_email = (args.user_email or bundle_email or "").strip().lower()
    if not target_email:
        raise RuntimeError(
            "Unable to infer user email from bundle. Pass --user-email explicitly."
        )

    files_index: dict[str, dict[str, Any]] = {}
    if files_manifest_path.is_file():
        for entry in _load_jsonl(files_manifest_path):
            file_id = entry.get("file_id")
            if file_id:
                files_index[file_id] = entry

    dry_run = args.dry_run
    split_branches = getattr(args, "split_branches", False)
    skip_branching = args.skip_branching and not split_branches
    upload_files = not args.no_upload_files
    api_base = (args.api_base or os.environ.get("ONYX_API_BASE") or "").rstrip("/")
    upload_session: requests.Session | None = None
    upload_url: str | None = None
    upload_timeout = float(os.environ.get("ONYX_UPLOAD_TIMEOUT_SECONDS", 120))
    if upload_files and not dry_run:
        if not api_base:
            raise RuntimeError(
                "Set ONYX_API_BASE (or pass --api-base) to enable attachment uploads."
            )
        upload_session = requests.Session()
        upload_session.headers.update(get_session_headers(content_type=None))
        upload_url = f"{api_base}/api/user/projects/file/upload"

    conn = None
    cursor = None
    if not dry_run:
        conn = _connect_postgres(args)
        cursor = conn.cursor()
        _ensure_metadata_table(cursor)

    user_id: str | None = None
    if cursor:
        cursor.execute('SELECT id FROM "user" WHERE email = %s', (target_email,))
        row = cursor.fetchone()
        if not row:
            raise RuntimeError(
                f"Onyx user with email {target_email} not found. Create user before importing."
            )
        user_id = row["id"] if isinstance(row, dict) else row[0]

    stats = {
        "processed": 0,
        "imported": 0,
        "skipped_branching": 0,
        "skipped_existing": 0,
        "dry_run_conversations": 0,
        "files_uploaded": 0,
        "files_missing": 0,
        "files_skipped": 0,
        "branches_split": 0,
    }

    uploaded_files: dict[str, dict[str, Any] | None] = {}

    def get_or_upload_file(file_id: str) -> dict[str, Any] | None:
        if file_id in uploaded_files:
            return uploaded_files[file_id]

        manifest = files_index.get(file_id)
        if not manifest:
            print(f"[warn] file_id {file_id} missing from files_manifest; skipping attachment")
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        if not upload_files:
            stats["files_skipped"] += 1
            uploaded_files[file_id] = None
            return None

        src_path = _resolve_file_source(bundle_path, manifest)
        if not src_path:
            print(f"[warn] file {file_id} has no local copy ({manifest.get('errors')}); skipping")
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        metadata = manifest.get("metadata") or {}
        file_name = metadata.get("filename") or metadata.get("name") or src_path.name
        content_type = (
            metadata.get("type")
            or metadata.get("content_type")
            or "application/octet-stream"
        )
        size_bytes = metadata.get("bytes")
        if not size_bytes:
            try:
                size_bytes = src_path.stat().st_size
            except OSError:
                size_bytes = None

        if not upload_session or not upload_url:
            raise RuntimeError("Attachment uploader session not initialized.")

        try:
            with src_path.open("rb") as fh:
                response = upload_session.post(
                    upload_url,
                    files=[("files", (file_name, fh, content_type))],
                    timeout=upload_timeout,
                )
        except requests.RequestException as exc:
            print(f"[error] Failed to upload {file_name} ({file_id}): {exc}")
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        if response.status_code >= 400:
            print(
                f"[error] Upload rejected for {file_name} ({file_id}): "
                f"{response.status_code} {response.text}"
            )
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        try:
            payload = response.json()
        except ValueError:
            print(f"[error] Unexpected upload response for {file_id}: {response.text}")
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        user_files_payload = payload.get("user_files") or []
        if not user_files_payload:
            skipped = payload.get("non_accepted_files") or []
            unsupported = payload.get("unsupported_files") or []
            if skipped or unsupported:
                print(
                    f"[warn] Onyx rejected {file_id} "
                    f"(non_accepted={skipped}, unsupported={unsupported})"
                )
            else:
                print(f"[warn] Onyx upload returned no files for {file_id}")
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        snapshot = user_files_payload[0]
        user_file_id = str(snapshot.get("id"))
        if not user_file_id:
            print(f"[warn] Missing user_file.id in upload response for {file_id}")
            stats["files_missing"] += 1
            uploaded_files[file_id] = None
            return None

        chat_file_type = snapshot.get("chat_file_type")
        resolved_content_type = snapshot.get("file_type") or content_type
        info = {
            "file_id": snapshot.get("file_id") or user_file_id,
            "user_file_id": user_file_id,
            "name": snapshot.get("name") or file_name,
            "content_type": resolved_content_type,
            "chat_file_type": chat_file_type
            or _guess_chat_file_type(resolved_content_type),
            "size_bytes": size_bytes,
            "librechat_file_id": file_id,
        }
        uploaded_files[file_id] = info
        stats["files_uploaded"] += 1
        return info

    max_convos = args.max_conversations if args.max_conversations else None

    for record in _load_jsonl(conversations_path):
        if max_convos is not None and stats["processed"] >= max_convos:
            break
        stats["processed"] += 1

        conversation = record.get("conversation") or {}
        messages = record.get("messages") or []
        branch_info = record.get("branching") or {}
        has_branching = bool(branch_info.get("has_branching"))

        if has_branching and skip_branching:
            stats["skipped_branching"] += 1
            continue

        conversation_uuid_obj = _safe_uuid(conversation.get("conversationId"))
        conversation_uuid_str = str(conversation_uuid_obj)
        description = (
            conversation.get("title")
            or conversation.get("greeting")
            or f"Imported chat {conversation_uuid_str}"
        )
        created_at = _parse_iso_datetime(conversation.get("createdAt"))
        updated_at = _parse_iso_datetime(conversation.get("updatedAt"))
        deleted = bool(conversation.get("isArchived"))
        shared_status = "PUBLIC" if conversation.get("shared") else "PRIVATE"
        persona_id = args.persona_id if args.persona_id is not None else 0
        current_alternate_model = conversation.get("model")
        temperature_override = conversation.get("temperature")
        branch_variants: list[list[dict[str, Any]]] = [messages]
        branch_context = False
        if has_branching and split_branches:
            split_variants = _split_messages_into_branches(messages)
            if split_variants:
                branch_variants = split_variants
                if len(split_variants) > 1:
                    branch_context = True
                    stats["branches_split"] += len(split_variants)
            else:
                print(
                    "[warn] Unable to derive branches for conversation "
                    f"{conversation.get('conversationId')}; importing full thread instead."
                )

        for idx, variant_messages in enumerate(branch_variants, start=1):
            is_branch = branch_context
            branch_index = idx if branch_context else None
            branch_total = len(branch_variants) if branch_context else None
            if branch_context:
                variant_uuid_obj = uuid.uuid5(
                    conversation_uuid_obj, f"branch-{idx}"
                )
                description_suffix = f" (branch #{idx})"
            else:
                variant_uuid_obj = conversation_uuid_obj
                description_suffix = ""
            variant_uuid_str = str(variant_uuid_obj)

            llm_override = _chat_metadata_payload(conversation)
            prompt_override = {
                "librechat": {
                    "promptPrefix": conversation.get("promptPrefix"),
                    "greeting": conversation.get("greeting"),
                    "tags": conversation.get("tags"),
                    "web_search": conversation.get("web_search"),
                    "temperature": conversation.get("temperature"),
                    "top_p": conversation.get("top_p"),
                }
            }
            if branch_context:
                llm_override.setdefault("librechat", {})["branch"] = {
                    "index": branch_index,
                    "total": branch_total,
                }
                prompt_override.setdefault("librechat", {})["branch"] = {
                    "index": branch_index,
                    "total": branch_total,
                }

            if dry_run:
                stats["dry_run_conversations"] += 1
                label = (
                    f"{variant_uuid_str} (branch #{branch_index})"
                    if is_branch
                    else variant_uuid_str
                )
                print(
                    f"[dry-run] Would import conversation {label} "
                    f"({len(variant_messages)} messages, created {created_at.isoformat()})"
                )
                continue

            cursor.execute(
                "SELECT 1 FROM chat_session WHERE id = %s",
                (variant_uuid_str,),
            )
            if cursor.fetchone():
                stats["skipped_existing"] += 1
                continue

            cursor.execute(
                """
                INSERT INTO chat_session (
                    id, user_id, description, deleted, shared_status,
                    time_created, time_updated, persona_id, llm_override,
                    prompt_override, onyxbot_flow, current_alternate_model,
                    temperature_override, project_id
                ) VALUES (
                    %(id)s, %(user_id)s, %(description)s, %(deleted)s, %(shared_status)s,
                    %(time_created)s, %(time_updated)s, %(persona_id)s, %(llm_override)s,
                    %(prompt_override)s, %(onyxbot_flow)s, %(current_alternate_model)s,
                    %(temperature_override)s, %(project_id)s
                )
                """,
                {
                    "id": variant_uuid_str,
                    "user_id": user_id,
                    "description": description + description_suffix,
                    "deleted": deleted,
                    "shared_status": shared_status,
                    "time_created": created_at,
                    "time_updated": updated_at,
                    "persona_id": persona_id,
                    "llm_override": json.dumps(llm_override),
                    "prompt_override": json.dumps(prompt_override),
                    "onyxbot_flow": False,
                    "current_alternate_model": current_alternate_model,
                    "temperature_override": temperature_override,
                    "project_id": None,
                },
            )

            session_intro = (
                f"Imported from LibreChat conversation {conversation.get('conversationId')} "
                f"(endpoint={conversation.get('endpoint')}, model={conversation.get('model')})."
            )
            if is_branch:
                session_intro += f" Branch #{branch_index} of {branch_total}."
            cursor.execute(
                """
                INSERT INTO chat_message (
                    message, message_type, time_sent, token_count,
                    parent_message, chat_session_id, citations, files,
                    error, rephrased_query, alternate_assistant_id,
                    overridden_model, is_agentic
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    session_intro,
                    "SYSTEM",
                    created_at,
                    0,
                    None,
                    variant_uuid_str,
                    json.dumps({}),
                    None,
                    None,
                    None,
                    None,
                    conversation.get("model"),
                    False,
                ),
            )
            row = cursor.fetchone()
            system_message_id = row["id"] if isinstance(row, dict) else row[0]

            message_id_map: dict[str, int] = {}
            previous_id = system_message_id

            for message in variant_messages:
                message_id = message.get("messageId")
                time_sent = _parse_iso_datetime(message.get("createdAt"))
                message_type = "USER" if message.get("isCreatedByUser") else "ASSISTANT"
                text = _message_text_from_record(message)
                if not text:
                    text = "(empty message)"
                token_count = int(message.get("tokenCount") or 0)
                parent_ext_id = message.get("parentMessageId")
            parent_id = None
            if (
                parent_ext_id
                and parent_ext_id != "00000000-0000-0000-0000-000000000000"
                and parent_ext_id in message_id_map
            ):
                parent_id = message_id_map[parent_ext_id]
            else:
                parent_id = previous_id
            metadata_blob = _message_metadata_payload(message)
            citations_payload: dict[str, Any] = {}
            message_files_payload: list[dict[str, Any]] = []
            if upload_files:
                file_sources: list[str] = []
                for field in ("files", "attachments"):
                    for entry in message.get(field) or []:
                        file_identifier = (
                            entry.get("file_id")
                            or entry.get("fileId")
                            or entry.get("id")
                        )
                        if file_identifier:
                            file_sources.append(file_identifier)

                for file_identifier in file_sources:
                    uploaded = get_or_upload_file(file_identifier)
                    if not uploaded:
                        continue
                    message_files_payload.append(
                        {
                            "id": uploaded["user_file_id"],
                            "name": uploaded["name"],
                            "type": uploaded.get("chat_file_type")
                            or _guess_chat_file_type(uploaded.get("content_type")),
                            "user_file_id": uploaded["user_file_id"],
                            "file_id": uploaded["file_id"],
                            "librechat_file_id": uploaded["librechat_file_id"],
                        }
                    )

            cursor.execute(
                """
                INSERT INTO chat_message (
                    message, message_type, time_sent, token_count,
                    parent_message, chat_session_id, citations, files,
                    error, rephrased_query, alternate_assistant_id,
                    overridden_model, is_agentic
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                (
                    text,
                    message_type,
                    time_sent,
                    token_count,
                    parent_id,
                    variant_uuid_str,
                    json.dumps(citations_payload),
                    json.dumps(message_files_payload) if message_files_payload else None,
                    str(message.get("error")) if message.get("error") else None,
                    None,
                    None,
                    message.get("model"),
                    False,
                ),
            )
            row = cursor.fetchone()
            inserted_id = row["id"] if isinstance(row, dict) else row[0]
            cursor.execute(
                """
                INSERT INTO librechat_message_metadata (chat_message_id, metadata)
                VALUES (%s, %s)
                ON CONFLICT (chat_message_id) DO UPDATE SET metadata = EXCLUDED.metadata
                """,
                (inserted_id, json.dumps(metadata_blob)),
            )
            message_id_map[message_id] = inserted_id
            previous_id = inserted_id
            if parent_id:
                cursor.execute(
                    "UPDATE chat_message SET latest_child_message = %s WHERE id = %s",
                    (inserted_id, parent_id),
                )

            stats["imported"] += 1
            if conn:
                conn.commit()

    if cursor:
        cursor.close()
    if conn:
        conn.close()

    print("Import summary:")
    for key, value in stats.items():
        print(f"  {key}: {value}")


def repair_latest_child(args):
    load_env(".env")
    conn = _connect_postgres(args)
    cursor = conn.cursor()

    session_ids: list[str] = []
    if args.chat_session_id:
        session_ids.append(args.chat_session_id)
    elif args.user_email:
        cursor.execute(
            """
            SELECT cs.id
            FROM chat_session cs
            JOIN "user" u ON u.id = cs.user_id
            WHERE lower(u.email) = lower(%s)
            """,
            (args.user_email,),
        )
        session_ids = [
            row["id"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()
        ]
        if not session_ids:
            raise RuntimeError(
                f"No chat sessions found for user {args.user_email}. "
                "Make sure the user exists and has imported chats."
            )
    elif args.all_sessions:
        cursor.execute("SELECT id FROM chat_session")
        session_ids = [
            row["id"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()
        ]
    else:
        raise RuntimeError(
            "Specify --chat-session-id, --user-email, or --all-sessions for repair."
        )

    repaired = 0
    for session_id in session_ids:
        cursor.execute(
            """
            SELECT id, parent_message
            FROM chat_message
            WHERE chat_session_id = %s
            ORDER BY time_sent ASC, id ASC
            """,
            (session_id,),
        )
        rows = cursor.fetchall()
        parent_children: dict[int, list[int]] = {}
        for row in rows:
            parent_id = row["parent_message"] if isinstance(row, dict) else row[1]
            child_id = row["id"] if isinstance(row, dict) else row[0]
            if parent_id:
                parent_children.setdefault(parent_id, []).append(child_id)

        for parent_id, children in parent_children.items():
            cursor.execute(
                "UPDATE chat_message SET latest_child_message = %s WHERE id = %s",
                (children[-1], parent_id),
            )
        if parent_children:
            repaired += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Updated latest_child_message for {repaired} chat sessions.")


def repair_citations(args):
    load_env(".env")
    conn = _connect_postgres(args)
    cursor = conn.cursor()

    filter_clause = ""
    filter_params: list[Any] = []

    if args.chat_session_id:
        filter_clause = "WHERE id = %s"
        filter_params.append(args.chat_session_id)
    elif args.user_email:
        filter_clause = (
            "WHERE user_id IN (SELECT id FROM \"user\" WHERE lower(email) = lower(%s))"
        )
        filter_params.append(args.user_email)
    elif not args.all_sessions:
        cursor.close()
        conn.close()
        raise RuntimeError(
            "Specify --chat-session-id, --user-email, or --all-sessions for repair-citations."
        )

    cursor.execute(
        f"SELECT id FROM chat_session {filter_clause}",
        filter_params,
    )
    session_ids = [row["id"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()]
    if not session_ids:
        print("No chat sessions matched the provided filters.")
        cursor.close()
        conn.close()
        return

    total_messages = 0
    for chunk_start in range(0, len(session_ids), 500):
        chunk = session_ids[chunk_start : chunk_start + 500]
        cursor.execute(
            """
            UPDATE chat_message
            SET citations = '{}'::jsonb
            WHERE chat_session_id = ANY(%s)
              AND citations IS NOT NULL
              AND jsonb_typeof(citations) <> 'object'
            RETURNING id
            """,
            (chunk,),
        )
        total_messages += cursor.rowcount or 0

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Sanitized citations for {total_messages} messages across {len(session_ids)} sessions.")


def repair_message_text(args):
    load_env(".env")
    conn = _connect_postgres(args)
    cursor = conn.cursor()

    filter_clause = ""
    filter_params: list[Any] = []

    if args.chat_session_id:
        filter_clause = "WHERE id = %s"
        filter_params.append(args.chat_session_id)
    elif args.user_email:
        filter_clause = (
            "WHERE user_id IN (SELECT id FROM \"user\" WHERE lower(email) = lower(%s))"
        )
        filter_params.append(args.user_email)
    elif not args.all_sessions:
        cursor.close()
        conn.close()
        raise RuntimeError(
            "Specify --chat-session-id, --user-email, or --all-sessions for repair-message-text."
        )

    cursor.execute(
        f"SELECT id FROM chat_session {filter_clause}",
        filter_params,
    )
    session_ids = [row["id"] if isinstance(row, dict) else row[0] for row in cursor.fetchall()]
    if not session_ids:
        print("No chat sessions matched the provided filters.")
        cursor.close()
        conn.close()
        return

    updated = 0
    for chunk_start in range(0, len(session_ids), 200):
        chunk = session_ids[chunk_start : chunk_start + 200]
        cursor.execute(
            """
            SELECT cm.id, cm.message, lmm.metadata
            FROM chat_message cm
            JOIN librechat_message_metadata lmm
              ON lmm.chat_message_id = cm.id
            WHERE cm.chat_session_id = ANY(%s)
              AND cm.message LIKE '%%[think]%%'
            """,
            (chunk,),
        )
        rows = cursor.fetchall()
        for row in rows:
            metadata = row["metadata"]
            if metadata is None:
                continue
            if isinstance(metadata, str):
                metadata = json.loads(metadata)
            rebuilt = _message_text_from_record(metadata)
            if rebuilt and rebuilt != row["message"]:
                cursor.execute(
                    "UPDATE chat_message SET message = %s WHERE id = %s",
                    (rebuilt, row["id"]),
                )
                updated += 1

    conn.commit()
    cursor.close()
    conn.close()
    print(f"Rebuilt message text for {updated} messages across {len(session_ids)} sessions.")


def _connect_postgres(args) -> "psycopg.Connection":
    _require_psycopg()
    host = args.db_host or os.environ.get("POSTGRES_HOST") or "localhost"
    port = int(args.db_port or os.environ.get("POSTGRES_PORT") or 5432)
    user = args.db_user or os.environ.get("POSTGRES_USER") or "postgres"
    password = args.db_password or os.environ.get("POSTGRES_PASSWORD") or "password"
    dbname = args.db_name or os.environ.get("POSTGRES_DB") or "postgres"
    conn = psycopg.connect(
        host=host,
        port=port,
        user=user,
        password=password,
        dbname=dbname,
        autocommit=False,
        row_factory=dict_row if dict_row else None,
    )
    return conn


def main():
    parser = argparse.ArgumentParser(description="LibreChat -> Onyx migration helper")
    sub = parser.add_subparsers(dest="command")

    create_cmd = sub.add_parser(
        "create-users", help="Create users from LibreChat MongoDB"
    )
    create_cmd.add_argument(
        "--skip-email",
        action="append",
        default=["philip.lilius@friendsagenda.se"],
        help="Email(s) to skip (can pass multiple)",
    )
    create_cmd.add_argument(
        "--output",
        help="Path to log file with generated credentials (default: created_users_TIMESTAMP.jsonl)",
    )

    export_cmd = sub.add_parser(
        "export-chats", help="Export LibreChat conversations into JSONL intermediates"
    )
    export_cmd.add_argument(
        "--user-email",
        action="append",
        required=True,
        help="Email(s) to export (repeat flag for multiple users)",
    )
    export_cmd.add_argument(
        "--output-dir",
        default="exports",
        help="Directory where export_<timestamp> folders will be created",
    )
    export_cmd.add_argument(
        "--asset-root",
        action="append",
        help="Additional directories to search for LibreChat file uploads/images",
    )
    export_cmd.add_argument(
        "--copy-files",
        action="store_true",
        help="Copy referenced files into the export bundle",
    )
    export_cmd.add_argument(
        "--max-conversations",
        type=int,
        help="Limit number of conversations per user (for dry runs/testing)",
    )

    import_cmd = sub.add_parser(
        "import-chats", help="Import LibreChat exports into Onyx Postgres"
    )
    import_cmd.add_argument(
        "--bundle-path",
        required=True,
        help="Path to the exported user folder (e.g., exports/export_TS/email-slug)",
    )
    import_cmd.add_argument(
        "--user-email",
        help="Override user email (defaults to bundle summary email)",
    )
    import_cmd.add_argument(
        "--max-conversations",
        type=int,
        help="Limit number of conversations to import (debugging)",
    )
    import_cmd.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse bundle and log actions without touching database",
    )
    import_cmd.add_argument(
        "--skip-branching",
        action="store_true",
        default=True,
        help="Skip conversations that contain branching threads (default: true)",
    )
    import_cmd.add_argument(
        "--split-branches",
        action="store_true",
        help="Import branched LibreChat conversations as separate Onyx sessions.",
    )
    import_cmd.add_argument(
        "--persona-id",
        type=int,
        help="Override persona_id to assign to imported chat sessions (default: 0)",
    )
    import_cmd.add_argument("--db-host", help="Postgres host (default env POSTGRES_HOST)")
    import_cmd.add_argument("--db-port", help="Postgres port (default env POSTGRES_PORT)")
    import_cmd.add_argument("--db-name", help="Postgres database (default POSTGRES_DB)")
    import_cmd.add_argument("--db-user", help="Postgres user (default POSTGRES_USER)")
    import_cmd.add_argument(
        "--db-password", help="Postgres password (default POSTGRES_PASSWORD)"
    )
    import_cmd.add_argument(
        "--no-upload-files",
        action="store_true",
        help="Skip uploading/copying attachment files (messages will omit file references)",
    )
    import_cmd.add_argument(
        "--api-base",
        help="Override ONYX_API_BASE for attachment ingestion (default env var)",
    )

    repair_cmd = sub.add_parser(
        "repair-latest-child",
        help="Backfill chat_message.latest_child_message for existing sessions",
    )
    repair_cmd.add_argument(
        "--chat-session-id",
        help="Specific chat_session.id to repair (repeat command for multiples)",
    )
    repair_cmd.add_argument(
        "--user-email",
        help="Repair all sessions belonging to this user email",
    )
    repair_cmd.add_argument(
        "--all-sessions",
        action="store_true",
        help="Repair every chat_session in the database (use with caution)",
    )
    repair_cmd.add_argument("--db-host", help="Postgres host (default env POSTGRES_HOST)")
    repair_cmd.add_argument("--db-port", help="Postgres port (default env POSTGRES_PORT)")
    repair_cmd.add_argument("--db-name", help="Postgres database (default POSTGRES_DB)")
    repair_cmd.add_argument("--db-user", help="Postgres user (default POSTGRES_USER)")
    repair_cmd.add_argument(
        "--db-password", help="Postgres password (default POSTGRES_PASSWORD)"
    )

    repair_citations_cmd = sub.add_parser(
        "repair-citations",
        help=(
            "Remove legacy LibreChat metadata from chat_message.citations "
            "so sessions load correctly"
        ),
    )
    repair_citations_cmd.add_argument(
        "--chat-session-id",
        help="Specific chat_session.id to sanitize",
    )
    repair_citations_cmd.add_argument(
        "--user-email",
        help="Sanitize all sessions belonging to this user",
    )
    repair_citations_cmd.add_argument(
        "--all-sessions",
        action="store_true",
        help="Sanitize every chat session (use cautiously)",
    )
    repair_citations_cmd.add_argument("--db-host", help="Postgres host (default env POSTGRES_HOST)")
    repair_citations_cmd.add_argument("--db-port", help="Postgres port (default env POSTGRES_PORT)")
    repair_citations_cmd.add_argument("--db-name", help="Postgres database (default POSTGRES_DB)")
    repair_citations_cmd.add_argument("--db-user", help="Postgres user (default POSTGRES_USER)")
    repair_citations_cmd.add_argument(
        "--db-password", help="Postgres password (default POSTGRES_PASSWORD)"
    )

    repair_text_cmd = sub.add_parser(
        "repair-message-text",
        help="Rebuild chat_message.message from stored metadata (e.g., wrap think blocks)",
    )
    repair_text_cmd.add_argument(
        "--chat-session-id",
        help="Specific chat_session.id to rebuild",
    )
    repair_text_cmd.add_argument(
        "--user-email",
        help="Rebuild all sessions belonging to this user",
    )
    repair_text_cmd.add_argument(
        "--all-sessions",
        action="store_true",
        help="Rebuild every chat session (use cautiously)",
    )
    repair_text_cmd.add_argument("--db-host", help="Postgres host (default env POSTGRES_HOST)")
    repair_text_cmd.add_argument("--db-port", help="Postgres port (default env POSTGRES_PORT)")
    repair_text_cmd.add_argument("--db-name", help="Postgres database (default POSTGRES_DB)")
    repair_text_cmd.add_argument("--db-user", help="Postgres user (default POSTGRES_USER)")
    repair_text_cmd.add_argument(
        "--db-password", help="Postgres password (default POSTGRES_PASSWORD)"
    )

    args = parser.parse_args()
    if not args.command:
        parser.error("No command specified. Try 'create-users'.")

    load_env(".env")

    if args.command == "create-users":
        create_users(args)
    elif args.command == "export-chats":
        export_chats(args)
    elif args.command == "import-chats":
        import_chats(args)
    elif args.command == "repair-latest-child":
        repair_latest_child(args)
    elif args.command == "repair-citations":
        repair_citations(args)
    elif args.command == "repair-message-text":
        repair_message_text(args)


if __name__ == "__main__":
    main()
