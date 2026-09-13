"""Provision a local account/workspace without emailing credentials or exposing passwords."""

import argparse
from getpass import getpass

from backend.app.config import get_settings
from backend.app.workspaces.api import RegisterRequest
from backend.app.workspaces.auth import AccountStore


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--email", required=True)
    parser.add_argument("--workspace", required=True)
    args = parser.parse_args()
    password = getpass("Password (at least 12 characters): ")
    if password != getpass("Confirm password: "):
        parser.error("Passwords do not match")
    request = RegisterRequest(email=args.email, password=password, workspace_name=args.workspace)
    settings = get_settings()
    accounts = AccountStore(
        settings.workspaces_path / "accounts.sqlite3", settings.auth_session_seconds
    )
    user = accounts.register(request.email, request.password, request.workspace_name)
    print(f"Account created. Workspace ID: {accounts.workspaces(user)[0]['id']}")


if __name__ == "__main__":
    main()
