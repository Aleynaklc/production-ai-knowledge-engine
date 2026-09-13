"""Password accounts, hashed expiring sessions, and server-owned memberships."""

import hashlib
import hmac
import secrets
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from time import time
from uuid import uuid4

from backend.app.documents.library import DocumentUploadError


class AccountStore:
    def __init__(self, path: Path, session_seconds: int) -> None:
        self.path, self.session_seconds = path, session_seconds
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS accounts (
                    id TEXT PRIMARY KEY, email TEXT UNIQUE NOT NULL, salt BLOB NOT NULL, password BLOB NOT NULL);
                CREATE TABLE IF NOT EXISTS workspaces (id TEXT PRIMARY KEY, name TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS memberships (
                    user_id TEXT NOT NULL REFERENCES accounts(id), workspace_id TEXT NOT NULL REFERENCES workspaces(id),
                    role TEXT NOT NULL CHECK(role IN ('owner','editor','reader')), PRIMARY KEY(user_id,workspace_id));
                CREATE TABLE IF NOT EXISTS sessions (
                    token_hash TEXT PRIMARY KEY, user_id TEXT NOT NULL REFERENCES accounts(id), expires REAL NOT NULL);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def password_hash(password: str, salt: bytes) -> bytes:
        return hashlib.scrypt(password.encode(), salt=salt, n=2**14, r=8, p=1, dklen=32)

    def register(self, email: str, password: str, workspace_name: str) -> str:
        salt = secrets.token_bytes(16)
        hashed = self.password_hash(password, salt)
        user_id, workspace_id = str(uuid4()), str(uuid4())
        try:
            with self.connect() as db:
                db.execute("INSERT INTO accounts VALUES (?,?,?,?)", (user_id, email, salt, hashed))
                db.execute("INSERT INTO workspaces VALUES (?,?)", (workspace_id, workspace_name))
                db.execute("INSERT INTO memberships VALUES (?,?,'owner')", (user_id, workspace_id))
        except sqlite3.IntegrityError as error:
            raise DocumentUploadError(
                "account_exists", "An account with this email already exists.", 409
            ) from error
        return user_id

    def login(self, email: str, password: str) -> str:
        with self.connect() as db:
            row = db.execute("SELECT * FROM accounts WHERE email=?", (email,)).fetchone()
        hashed = self.password_hash(password, row["salt"] if row else b"\0" * 16)
        if row is None or not hmac.compare_digest(hashed, row["password"]):
            raise DocumentUploadError("invalid_credentials", "Email or password is incorrect.", 401)
        return str(row["id"])

    def new_session(self, user_id: str) -> str:
        token = secrets.token_urlsafe(32)
        with self.connect() as db:
            db.execute("DELETE FROM sessions WHERE expires<=?", (time(),))
            db.execute(
                "INSERT INTO sessions VALUES (?,?,?)",
                (
                    hashlib.sha256(token.encode()).hexdigest(),
                    user_id,
                    time() + self.session_seconds,
                ),
            )
        return token

    def user(self, token: str | None) -> dict[str, str]:
        digest = hashlib.sha256((token or "").encode()).hexdigest()
        with self.connect() as db:
            row = db.execute(
                "SELECT a.id,a.email FROM accounts a JOIN sessions s ON s.user_id=a.id WHERE s.token_hash=? AND s.expires>?",
                (digest, time()),
            ).fetchone()
        if row is None:
            raise DocumentUploadError("authentication_required", "Sign in to continue.", 401)
        return {"id": row[0], "email": row[1]}

    def logout(self, token: str | None) -> None:
        with self.connect() as db:
            db.execute(
                "DELETE FROM sessions WHERE token_hash=?",
                (hashlib.sha256((token or "").encode()).hexdigest(),),
            )

    def workspaces(self, user_id: str | None = None) -> list[dict[str, str]]:
        with self.connect() as db:
            if user_id is None:
                return [dict(row) for row in db.execute("SELECT * FROM workspaces")]
            return [
                dict(row)
                for row in db.execute(
                    "SELECT w.*,m.role FROM workspaces w JOIN memberships m ON w.id=m.workspace_id WHERE m.user_id=? ORDER BY w.name,w.id",
                    (user_id,),
                )
            ]

    def authorize(self, user_id: str, workspace_id: str, write: bool = False) -> str:
        with self.connect() as db:
            row = db.execute(
                "SELECT role FROM memberships WHERE user_id=? AND workspace_id=?",
                (user_id, workspace_id),
            ).fetchone()
        if row is None:
            raise DocumentUploadError("workspace_not_found", "Workspace not found.", 404)
        if write and row[0] == "reader":
            raise DocumentUploadError(
                "permission_denied", "This action requires an editor role.", 403
            )
        return str(row[0])

    def add_member(self, owner: str, workspace_id: str, email: str, role: str) -> None:
        if self.authorize(owner, workspace_id) != "owner":
            raise DocumentUploadError(
                "permission_denied", "Only the workspace owner can manage members.", 403
            )
        with self.connect() as db:
            user = db.execute("SELECT id FROM accounts WHERE email=?", (email,)).fetchone()
            if user is None:
                raise DocumentUploadError(
                    "account_not_found", "This person must create an account first.", 404
                )
            existing = db.execute(
                "SELECT role FROM memberships WHERE user_id=? AND workspace_id=?",
                (user[0], workspace_id),
            ).fetchone()
            if existing and existing[0] == "owner":
                raise DocumentUploadError(
                    "owner_protected", "The owner role cannot be changed here.", 409
                )
            db.execute(
                "INSERT INTO memberships VALUES (?,?,?) ON CONFLICT(user_id,workspace_id) DO UPDATE SET role=excluded.role",
                (user[0], workspace_id, role),
            )

    def remove_member(self, owner: str, workspace_id: str, member_id: str) -> None:
        if self.authorize(owner, workspace_id) != "owner":
            raise DocumentUploadError(
                "permission_denied", "Only the owner can remove members.", 403
            )
        with self.connect() as db:
            member = db.execute(
                "SELECT role FROM memberships WHERE user_id=? AND workspace_id=?",
                (member_id, workspace_id),
            ).fetchone()
            if member is not None and member[0] == "owner":
                raise DocumentUploadError(
                    "owner_protected", "The workspace owner cannot be removed here.", 409
                )
            db.execute(
                "DELETE FROM memberships WHERE user_id=? AND workspace_id=? AND role!='owner'",
                (member_id, workspace_id),
            )

    def members(self, workspace_id: str) -> list[dict[str, str]]:
        with self.connect() as db:
            return [
                dict(row)
                for row in db.execute(
                    "SELECT a.id,a.email,m.role FROM memberships m JOIN accounts a ON a.id=m.user_id WHERE m.workspace_id=? ORDER BY a.email",
                    (workspace_id,),
                )
            ]
