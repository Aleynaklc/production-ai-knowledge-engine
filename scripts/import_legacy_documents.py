"""Explicitly assign preserved legacy shared documents to one workspace (offline admin CLI)."""

import argparse
import json
import sqlite3
from contextlib import closing

from backend.app.config import get_settings
from backend.app.workspaces.auth import AccountStore
from backend.app.workspaces.store import WorkspaceStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--workspace-id", required=True, help="Destination ID from your account/workspace"
    )
    args = parser.parse_args()
    settings = get_settings()
    accounts = AccountStore(
        settings.workspaces_path / "accounts.sqlite3", settings.auth_session_seconds
    )
    if args.workspace_id not in {item["id"] for item in accounts.workspaces()}:
        parser.error("Unknown workspace; create the owner account first")
    if not settings.documents_path.exists():
        parser.error("Legacy document database does not exist")
    store = WorkspaceStore(
        settings.workspaces_path / args.workspace_id / "library.sqlite3", settings
    )
    with closing(
        sqlite3.connect(settings.documents_path.resolve().as_uri() + "?mode=ro", uri=True)
    ) as db:
        rows = db.execute(
            "SELECT record_json,content FROM documents ORDER BY created_at"
        ).fetchall()
    queued = 0
    for record_json, content in rows:
        record = json.loads(record_json)
        _, duplicate = store.enqueue(record["filename"], content.encode("utf-8"))
        queued += not duplicate
    print(
        f"Queued {queued} documents. Legacy storage was left unchanged. Start the API to process them."
    )


if __name__ == "__main__":
    main()
