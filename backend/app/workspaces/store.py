"""A separate durable SQLite library and job queue per authenticated workspace."""

import hashlib
import json
import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from pydantic import BaseModel

from backend.app.config import Settings
from backend.app.documents.validation import DocumentUploadError, _safe_filename
from backend.app.ingestion.models import DocumentChunk
from backend.app.workspaces.parsing import SourceUnit


def ingestion_profile(settings: Settings) -> str:
    """Invalidate derived chunks when parsing, tokenization, or chunking changes."""
    values = settings.model_dump(
        include={
            "chunk_strategy",
            "chunk_size_tokens",
            "chunk_overlap_tokens",
            "model_name",
            "model_revision",
            "embedding_model_name",
            "embedding_model_revision",
            "upload_ocr_enabled",
            "upload_ocr_languages",
        }
    )
    return hashlib.sha256(
        ("layout-structure-v2:" + json.dumps(values, sort_keys=True)).encode()
    ).hexdigest()


class WorkspaceDocument(BaseModel):
    id: str
    filename: str
    title: str
    size_bytes: int
    created_at: str
    version: int
    active_version: int | None
    status: str
    error: str | None
    chunk_count: int


class WorkspaceStore:
    def __init__(self, path: Path, settings: Settings) -> None:
        self.path = path
        self.settings = settings
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connect() as db:
            db.executescript("""
                PRAGMA journal_mode=WAL;
                CREATE TABLE IF NOT EXISTS documents (
                    id TEXT PRIMARY KEY, filename TEXT NOT NULL, created_at TEXT NOT NULL,
                    version INTEGER NOT NULL, active_version INTEGER, status TEXT NOT NULL,
                    error TEXT, deleted INTEGER NOT NULL DEFAULT 0);
                CREATE TABLE IF NOT EXISTS versions (
                    document_id TEXT NOT NULL REFERENCES documents(id), version INTEGER NOT NULL,
                    sha256 TEXT NOT NULL, filename TEXT NOT NULL, data BLOB NOT NULL, units TEXT NOT NULL DEFAULT '[]',
                    chunks TEXT NOT NULL DEFAULT '[]', vectors TEXT NOT NULL DEFAULT '[]',
                    PRIMARY KEY(document_id, version));
                CREATE TABLE IF NOT EXISTS jobs (
                    id INTEGER PRIMARY KEY, document_id TEXT NOT NULL, version INTEGER NOT NULL,
                    state TEXT NOT NULL DEFAULT 'queued');
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            """)

    @contextmanager
    def connect(self) -> Iterator[sqlite3.Connection]:
        db = sqlite3.connect(self.path, timeout=30)
        try:
            db.row_factory = sqlite3.Row
            db.execute("PRAGMA foreign_keys=ON")
            with db:
                yield db
        finally:
            db.close()

    def _record(self, row: sqlite3.Row) -> WorkspaceDocument:
        return WorkspaceDocument(
            id=row["id"],
            filename=row["filename"],
            title=Path(row["filename"]).stem,
            size_bytes=row["size_bytes"],
            created_at=row["created_at"],
            version=row["version"],
            active_version=row["active_version"],
            status=row["status"],
            error=row["error"],
            chunk_count=len(json.loads(row["chunks"])),
        )

    def documents(self) -> list[WorkspaceDocument]:
        with self.connect() as db:
            return [
                self._record(row)
                for row in db.execute("""
                SELECT d.*, LENGTH(v.data) AS size_bytes, COALESCE(a.chunks, '[]') AS chunks
                FROM documents d JOIN versions v ON v.document_id=d.id AND v.version=d.version
                LEFT JOIN versions a ON a.document_id=d.id AND a.version=d.active_version
                WHERE deleted=0 ORDER BY d.created_at DESC
            """)
            ]

    def get(self, document_id: str) -> WorkspaceDocument:
        record = next((item for item in self.documents() if item.id == document_id), None)
        if record is None:
            raise DocumentUploadError("document_not_found", "Document not found.", 404)
        return record

    def enqueue(
        self, filename: str, data: bytes, document_id: str | None = None
    ) -> tuple[WorkspaceDocument, bool]:
        # Reuse path/control-character validation without treating a binary format as text.
        suffix = Path(filename).suffix.lower()
        if suffix not in {".md", ".txt", ".pdf", ".docx"}:
            raise DocumentUploadError(
                "unsupported_extension", "Use PDF, DOCX, Markdown, or text.", 415
            )
        safe = _safe_filename(
            filename if suffix in {".md", ".txt"} else filename[: -len(suffix)] + ".txt"
        )
        filename = safe if suffix in {".md", ".txt"} else safe[:-4] + suffix
        if not data:
            raise DocumentUploadError("empty_document", "The file is empty.", 422)
        if len(data) > self.settings.upload_max_bytes:
            raise DocumentUploadError("file_too_large", "The file exceeds the upload limit.", 413)
        digest = hashlib.sha256(data).hexdigest()
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            if document_id is None:
                duplicate = db.execute(
                    """SELECT d.id FROM documents d JOIN versions v
                    ON v.document_id=d.id AND v.version=d.version
                    WHERE d.deleted=0 AND v.sha256=? AND d.status!='failed'""",
                    (digest,),
                ).fetchone()
                if duplicate:
                    return self.get(duplicate[0]), True
                count = db.execute("SELECT COUNT(*) FROM documents WHERE deleted=0").fetchone()[0]
                if count >= self.settings.upload_max_documents:
                    raise DocumentUploadError(
                        "document_limit_exceeded", "Workspace document limit reached.", 409
                    )
                document_id = str(uuid4())
                version = 1
                db.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, 1, NULL, 'processing', NULL, 0)",
                    (document_id, filename, datetime.now(UTC).isoformat()),
                )
            else:
                row = db.execute(
                    "SELECT * FROM documents WHERE id=? AND deleted=0", (document_id,)
                ).fetchone()
                if row is None:
                    raise DocumentUploadError("document_not_found", "Document not found.", 404)
                if row["status"] == "processing":
                    raise DocumentUploadError(
                        "document_busy", "Wait for processing to finish.", 409
                    )
                if row["version"] >= self.settings.upload_max_versions:
                    raise DocumentUploadError(
                        "version_limit_exceeded", "Document version limit reached.", 409
                    )
                version = row["version"] + 1
                db.execute(
                    "UPDATE documents SET version=?, filename=?, status='processing', error=NULL WHERE id=?",
                    (version, filename, document_id),
                )
            db.execute(
                "INSERT INTO versions(document_id,version,sha256,filename,data) VALUES (?,?,?,?,?)",
                (document_id, version, digest, filename, data),
            )
            db.execute("INSERT INTO jobs(document_id,version) VALUES (?,?)", (document_id, version))
            db.commit()
        return self.get(document_id), False

    def recover(self) -> None:
        with self.connect() as db:
            db.execute("UPDATE jobs SET state='queued' WHERE state='running'")
            db.commit()

    def schedule_reindex(self) -> int:
        """Queue versioned replacements, retaining the old ready evidence until publication."""
        profile = ingestion_profile(self.settings)
        with self.connect() as db:
            rows = db.execute(
                """SELECT d.id,v.filename,v.data,v.chunks FROM documents d
                JOIN versions v ON v.document_id=d.id AND v.version=d.active_version
                WHERE d.deleted=0 AND d.status='ready' AND d.version < ?""",
                (self.settings.upload_max_versions,),
            ).fetchall()
        queued = 0
        for row in rows:
            chunks = json.loads(row["chunks"])
            if chunks and all(
                chunk.get("metadata", {}).get("ingestion_profile") == profile for chunk in chunks
            ):
                continue
            try:
                self.enqueue(row["filename"], row["data"], row["id"])
                queued += 1
            except DocumentUploadError as error:
                if error.code != "document_busy":
                    raise
        return queued

    def next_job(self) -> tuple[int, str, int, str, bytes] | None:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute("""SELECT j.id, j.document_id, j.version, d.filename, v.data
                FROM jobs j JOIN documents d ON d.id=j.document_id
                JOIN versions v ON v.document_id=j.document_id AND v.version=j.version
                WHERE j.state='queued' AND d.deleted=0 ORDER BY j.id LIMIT 1""").fetchone()
            if not row:
                return None
            db.execute("UPDATE jobs SET state='running' WHERE id=?", (row[0],))
            db.commit()
            return (row[0], row[1], row[2], row[3], row[4])

    def snapshot(self) -> tuple[str, list[DocumentChunk], list[list[float]]]:
        with self.connect() as db:
            rows = db.execute("""SELECT v.chunks,v.vectors FROM versions v JOIN documents d
                ON d.id=v.document_id AND d.active_version=v.version WHERE d.deleted=0 ORDER BY d.id""").fetchall()
        chunks = [DocumentChunk.model_validate(item) for row in rows for item in json.loads(row[0])]
        vectors = [vector for row in rows for vector in json.loads(row[1])]
        revision = hashlib.sha256("".join(chunk.chunk_id for chunk in chunks).encode()).hexdigest()
        return revision, chunks, vectors

    def complete(
        self,
        job_id: int,
        document_id: str,
        version: int,
        units: list[SourceUnit],
        chunks: list[DocumentChunk],
        vectors: list[list[float]],
    ) -> bool:
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            row = db.execute(
                "SELECT version,deleted FROM documents WHERE id=?", (document_id,)
            ).fetchone()
            if row is None or row[1] or row[0] != version:
                return False
            db.execute(
                "UPDATE versions SET units=?,chunks=?,vectors=? WHERE document_id=? AND version=?",
                (
                    json.dumps([unit.model_dump() for unit in units]),
                    json.dumps([chunk.model_dump() for chunk in chunks]),
                    json.dumps(vectors),
                    document_id,
                    version,
                ),
            )
            db.execute(
                "UPDATE documents SET active_version=?, status='ready', error=NULL WHERE id=?",
                (version, document_id),
            )
            db.execute("UPDATE jobs SET state='done' WHERE id=?", (job_id,))
            db.commit()
            return True

    def fail(self, job_id: int, document_id: str, version: int, message: str) -> None:
        with self.connect() as db:
            db.execute(
                "UPDATE documents SET status='failed',error=? WHERE id=? AND version=? AND deleted=0",
                (message, document_id, version),
            )
            db.execute("UPDATE jobs SET state='failed' WHERE id=?", (job_id,))
            db.commit()

    def delete(self, document_id: str) -> None:
        self.get(document_id)
        with self.connect() as db:
            db.execute("BEGIN IMMEDIATE")
            db.execute("UPDATE documents SET deleted=1 WHERE id=?", (document_id,))
            db.execute("DELETE FROM jobs WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM versions WHERE document_id=?", (document_id,))
            db.execute("DELETE FROM documents WHERE id=?", (document_id,))
            db.commit()

    def source(self, document_id: str, version: int) -> tuple[str, bytes, list[SourceUnit]]:
        self.get(document_id)
        with self.connect() as db:
            row = db.execute(
                "SELECT data,units,filename FROM versions WHERE document_id=? AND version=?",
                (document_id, version),
            ).fetchone()
        if row is None or row[1] == "[]":
            raise DocumentUploadError("source_not_found", "Source version is not available.", 404)
        return row[2], row[0], [SourceUnit.model_validate(item) for item in json.loads(row[1])]

    def retry(self, document_id: str) -> WorkspaceDocument:
        record = self.get(document_id)
        if record.status != "failed":
            raise DocumentUploadError("document_busy", "Only failed documents can be retried.", 409)
        with self.connect() as db:
            db.execute(
                "UPDATE jobs SET state='queued' WHERE document_id=? AND version=?",
                (document_id, record.version),
            )
            db.execute(
                "UPDATE documents SET status='processing',error=NULL WHERE id=?", (document_id,)
            )
        return self.get(document_id)
