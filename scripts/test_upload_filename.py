#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any

import requests


def load_env(path: Path) -> None:
    if not path.is_file():
        return
    for raw in path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if not key or key in os.environ:
            continue
        os.environ[key] = value.strip().strip('"').strip("'")


def resolve_file_path(bundle_root: Path, entry: dict[str, Any]) -> Path | None:
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


def pick_entry(manifest_path: Path, start_index: int) -> tuple[dict[str, Any], Path]:
    with manifest_path.open() as handle:
        for idx, line in enumerate(handle):
            if idx < start_index:
                continue
            if not line.strip():
                continue
            entry = json.loads(line)
            candidate = resolve_file_path(manifest_path.parent, entry)
            if candidate:
                return entry, candidate
    raise RuntimeError("No local file copies found in manifest after index.")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Upload one file from files_manifest.jsonl and report the stored filename."
    )
    parser.add_argument(
        "--manifest",
        default="files_manifest.jsonl",
        help="Path to files_manifest.jsonl (defaults to current directory).",
    )
    parser.add_argument(
        "--index",
        type=int,
        default=0,
        help="Start at this line index when scanning the manifest.",
    )
    parser.add_argument(
        "--api-base",
        default=None,
        help="ONYX API base URL (or set ONYX_API_BASE).",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="ONYX API key (or set ONYX_API_KEY).",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Upload timeout in seconds.",
    )
    args = parser.parse_args()

    load_env(Path(".env"))

    api_base = args.api_base or os.environ.get("ONYX_API_BASE")
    api_key = args.api_key or os.environ.get("ONYX_API_KEY")
    if not api_base or not api_key:
        raise RuntimeError("Set ONYX_API_BASE and ONYX_API_KEY (or pass --api-base/--api-key).")

    manifest_path = Path(args.manifest).expanduser()
    if not manifest_path.is_file():
        raise RuntimeError(f"Manifest not found: {manifest_path}")

    entry, file_path = pick_entry(manifest_path, args.index)
    metadata = entry.get("metadata") or {}
    file_name = metadata.get("filename") or metadata.get("name") or file_path.name
    content_type = (
        metadata.get("type")
        or metadata.get("content_type")
        or "application/octet-stream"
    )

    headers = {"Authorization": f"Bearer {api_key}"}
    upload_url = f"{api_base.rstrip('/')}/api/user/projects/file/upload"

    with file_path.open("rb") as fh:
        response = requests.post(
            upload_url,
            headers=headers,
            files=[("files", (file_name, fh, content_type))],
            timeout=args.timeout,
        )

    print(f"HTTP {response.status_code}")
    if response.status_code >= 400:
        print(response.text)
        return 1

    payload = response.json()
    user_files = payload.get("user_files") or []
    if not user_files:
        print(payload)
        return 1

    uploaded = user_files[0]
    print(
        json.dumps(
            {
                "expected_name": file_name,
                "uploaded_name": uploaded.get("name"),
                "user_file_id": uploaded.get("id"),
                "file_id": uploaded.get("file_id"),
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
