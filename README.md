LibreChat → Onyx Migrator
=========================

This utility repo contains small Python helpers that support the LibreChat → Onyx migration workstream.  
All scripts are read-only with respect to LibreChat/Onyx data—only new export artifacts are created on disk.

Made with Codex, by OpenAI  
https://github.com/openai/codex

Environment
-----------
1. Copy `.env.example` to `.env` (or reuse an existing file) so the following variables are available:
   - `LIBRECHAT_MONGODB` – Mongo connection string (e.g. `mongodb://127.0.0.1:27018/LibreChat`).  
     If the default hostname `mongodb.librechat.local` is unreachable from your shell, override this env var when running commands.
   - `ONYX_API_BASE` / `ONYX_API_KEY` – required for the user-creation flow **and** for attachment ingestion during imports (files are uploaded via the Onyx API).
2. Optional: provide extra attachment roots so the exporter can locate binary files:
   - `LIBRECHAT_ASSET_ROOTS=/path/to/LibreChat/uploads:/path/to/LibreChat/images`
   - or pass `--asset-root` flags to the exporter.

Commands
--------
### 1. Create missing Onyx users
```
python lc-mig.py create-users \
  --skip-email example.user@company.com \
  --output ./created_users.jsonl
```
The script:
- fetches current Onyx accounts via `/api/manage/users`
- iterates LibreChat’s `users` collection
- POSTs new Onyx accounts for anyone missing, logging temporary credentials to the JSONL file you specify.

### 2. Export LibreChat chat history (intermediary files)
```
python lc-mig.py export-chats \
  --user-email example.user@company.com \
  --output-dir ./exports \
  --asset-root /path/to/LibreChat/uploads \
  --asset-root /path/to/LibreChat/images \
  --copy-files
```

Key flags:
- `--user-email` (repeatable, required): which LibreChat users to export.
- `--output-dir`: parent directory where `export_<timestamp>/` folders are generated (defaults to `./exports`).
- `--asset-root`: additional directories that might contain referenced uploads or generated images. Defaults cover `../LibreChat/{uploads,images}`.
- `--copy-files`: if set, binary attachments are copied into the export bundle under `files/<file_id>/`.
- `--max-conversations`: optional limiter for dry runs.

Output layout per user:
```
exports/export_YYYYmmddHHMMSS/<email-slug>/
  summary.json             # counts, timestamps, asset roots searched
  conversations.jsonl      # 1 JSON object per conversation (metadata + ordered messages)
  files_manifest.jsonl     # deduplicated file metadata + sha256 + copy status
  files/                   # optional copies when --copy-files is used
```

Each `conversations.jsonl` record contains:
- normalized conversation doc from Mongo (`conversation` key)
- linearized messages ordered by `createdAt`
- detected branching info (`branch_points`, `missing_parent_ids`)
- referenced file ids + any missing ids

### 3. Import bundle into Onyx (beta)
> :warning: Requires `psycopg[binary]` and network access to both Postgres (relational_db) and the Onyx API host (for `/api/user/projects/file/upload`). Attachments are uploaded automatically unless you pass `--no-upload-files`.

```
pip install -r requirements.txt

python lc-mig.py import-chats \
  --bundle-path exports/export_YYYYmmddHHMMSS/example-user-company-com \
  --dry-run
```

Drop `--dry-run` once satisfied. Key flags:
- `--db-host/port/user/password` override the Postgres connection (defaults pull from env).
- `--api-base` overrides `ONYX_API_BASE` if you need to point at a different Onyx host for attachment ingestion.
- `--no-upload-files` skips attachment handling if you only care about text.
- `--skip-branching` (default true) ignores conversations that contain branching message trees. Remove it once you’re ready to split/duplicate those chats.

The importer currently:
- inserts a synthetic SYSTEM message per chat to capture LibreChat metadata in `llm_override` / `prompt_override`.
- alternates USER/ASSISTANT messages in chronological order, embedding the original LibreChat payload in `citations`.
- uploads referenced files via `/api/user/projects/file/upload`, which creates the same `user_file`/`file_record` rows the UI expects, so `chat_message.files` mirrors the original attachments.

Batch exporting + importing every user
--------------------------------------
1. **Create the email list once.** Pull it straight from Mongo so you can reuse it for retries:
   ```
   python - <<'PY'
   import os
   from pymongo import MongoClient

   mongo_uri = os.environ.get("LIBRECHAT_MONGODB", "mongodb://mongodb.librechat.local:27017/")
   client = MongoClient(mongo_uri, connect=False)
   db = client["LibreChat"]
   with open("all_users.txt", "w") as handle:
       for doc in db.users.find({}, {"email": 1}):
           email = (doc.get("email") or "").strip()
           if email:
               handle.write(email + "\n")
   print("Wrote", sum(1 for _ in open("all_users.txt")), "emails to all_users.txt")
   PY
   ```

2. **Export in manageable batches.** `export-chats` accepts multiple `--user-email` flags, so split the list (example: 25 users per batch) and pass each chunk in one invocation:
   ```
   split -l 25 all_users.txt batch_
   for file in batch_*; do
     python lc-mig.py export-chats \
       $(sed "s/^/--user-email /" "$file") \
       --output-dir exports \
       --asset-root /path/to/LibreChat/uploads \
       --asset-root /path/to/LibreChat/images \
       --copy-files
   done
   ```
   Every run creates `exports/export_<timestamp>/` with one subdirectory per user (summary + JSONL + optional files/).

3. **Import all bundles.** After a spot-check, loop through each user folder and run the importer (dry run first if you want a preview):
   ```
   find exports -mindepth 2 -maxdepth 2 -type d -print0 |
     while IFS= read -r -d '' bundle; do
       python lc-mig.py import-chats --bundle-path "$bundle" --dry-run
     done
   ```
   Drop `--dry-run` once satisfied. If you need to re-import a user, purge their existing Onyx sessions first (except any you deliberately preserve) so you don’t create duplicates.

A note on older imports: sessions ingested before this version may have empty threads in the UI because `latest_child_message` wasn’t populated. Run the repair command once per user (or globally) to backfill pointers:
```
python lc-mig.py repair-latest-child --user-email example.user@company.com
```
Use `--chat-session-id <uuid>` for targeted fixes or `--all-sessions` if you truly need to sweep everything.

A fully featured importer (persona remapping, smarter branching, etc.) will build on this foundation; tracking issue: `librechat-migrator-9nx`.

Expose Postgres Port
--------------------
If you prefer running the importer from outside Docker, expose Postgres via a small override so `relational_db` listens on the host network. (Replace `localhost` in the example below with whatever hostname/IP your environment requires.)

`deployment/docker-compose.override.yml`
```yaml
services:
  relational_db:
    ports:
      - "5432:5432"
```

Then restart the service:
```
cd deployment
docker compose up -d relational_db
```

Smoke-test the connection:
```
HOST_IP=<server-ip-or-docker-hostname>
psql postgres://postgres:password@$HOST_IP:5432/postgres -c '\dt chat_session'
```

With that in place you can run:
```
python lc-mig.py import-chats \
  --bundle-path exports/export_YYYYmmddHHMMSS/example-user-company-com \
  --db-host $HOST_IP --db-port 5432
```
Remember to tear down the override (or firewall the port) when you’re finished.

Maintenance commands
--------------------
- `python lc-mig.py repair-latest-child --user-email <email>` backfills `latest_child_message` pointers for sessions imported before the fix.
- `python lc-mig.py repair-citations --user-email <email>` removes the legacy LibreChat metadata blob we previously stored in `chat_message.citations` (which causes the 400 error in `/chat/get-chat-session`). Use `--all-sessions` to sweep everything if needed.

Metadata Strategy
-----------------
- Export artifacts already namespace LibreChat-specific info so we can later insert it into Onyx’s `chat_session.prompt_override.librechat` and related JSON columns.
- Longer term we can replay the same JSON into dedicated `chat_session_metadata` / `chat_message_metadata` tables for first-class querying.
- Branching conversations are flagged during export so downstream tooling can decide whether to split them into separate Onyx sessions (e.g. append “(branched from <conversationId>)” to the title).

Operational Notes
-----------------
- These scripts never delete or mutate LibreChat data; they only read via MongoDB and (optionally) copy files locally.
- Always run exports against low-risk users first (e.g. `example.user@company.com`) before batching the rest.
- Ensure the Onyx containers are running if you plan to reuse the resulting JSON with future import tooling (they provide the API + LLM configs referenced in planning).

Development
-----------
- Install linting tools with `pip install ruff`.
- Run `ruff check lc-mig.py` (optionally `--fix`) before committing to keep the single-file CLI tidy.
