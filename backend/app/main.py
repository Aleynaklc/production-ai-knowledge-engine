"""Production entry point: authenticated workspace API only."""

from backend.app.workspaces.api import create_app

__all__ = ["app", "create_app"]

app = create_app()
