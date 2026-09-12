"""Atomic, bounded persistence for uploaded UTF-8 Markdown and text documents."""

import hashlib
import re
import sqlite3
import unicodedata
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from threading import Lock

from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.documents.models import DocumentRecord, UploadResult
from backend.app.ingestion.chunkers import (
    FixedTokenChunker,
    HuggingFaceTokenCodec,
    RecursiveChunker,
    TokenCodec,
)
from backend.app.ingestion.cleaner import clean_text
from backend.app.ingestion.models import DocumentChunk, RawDocument
from backend.app.ingestion.pipeline import build_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class DocumentUploadError(ValueError):
    """An actionable upload failure that the API can expose without internal paths."""

    def __init__(self, code: str, message: str, status_code: int) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.status_code = status_code


def _safe_filename(filename: str) -> str:
    if (
        not filename
        or "/" in filename
        or "\\" in filename
        or any(unicodedata.category(character).startswith("C") for character in filename)
    ):
        raise DocumentUploadError("invalid_filename", "Use a filename without a path.", 400)
    normalized = unicodedata.normalize("NFC", filename).strip()
    if not normalized or len(normalized.encode("utf-8")) > 255:
        raise DocumentUploadError("invalid_filename", "Filename must be 1–255 UTF-8 bytes.", 400)
    suffix = Path(normalized).suffix.lower()
    if suffix not in {".md", ".txt"}:
        raise DocumentUploadError(
            "unsupported_extension", "Only UTF-8 .md and .txt documents are supported.", 415
        )
    # The name is citation/display metadata only; it is never used as a disk path.
    stem = re.sub(r"[^\w .()-]", "_", Path(normalized).stem, flags=re.UNICODE).strip(" .")
    if not stem:
        raise DocumentUploadError("invalid_filename", "Filename must include a document name.", 400)
    return stem + suffix


def _decode_document(data: bytes) -> str:
    try:
        text = data.decode("utf-8-sig")
    except UnicodeDecodeError as error:
        raise DocumentUploadError(
            "invalid_encoding", "Save the document as UTF-8 text before uploading.", 422
        ) from error
    if any(
        unicodedata.category(character) == "Cc" and character not in "\n\r\t" for character in text
    ):
        raise DocumentUploadError(
            "binary_document",
            "The document contains binary or unsupported control characters.",
            422,
        )
    cleaned = clean_text(text)
    if not cleaned or not any(not character.isspace() for character in cleaned):
        raise DocumentUploadError("empty_document", "The document has no text to index.", 422)
    return cleaned


def _title(filename: str, text: str) -> str:
    title = Path(filename).stem.replace("_", " ")
    if filename.endswith(".md"):
        for line in text.splitlines():
            prefix = re.match(r"^ {0,3}#{1,6}[ \t]+", line)
            if prefix:
                heading = line[prefix.end() :].strip().rstrip("#").strip()
                if heading:
                    title = heading
                    break
    # Titles are copied to every chunk and rendered in the model's source header.
    # Keep metadata bounded while preserving the full heading in the document body.
    title = " ".join(title.split())
    encoded = title.encode("utf-8")
    return (
        encoded[:157].decode("utf-8", errors="ignore").rstrip() + "…"
        if len(encoded) > 160
        else title
    )


class DocumentLibrary:
    """Store documents and chunks together, sharing no SQLite handles across threads."""

    def __init__(
        self,
        settings: Settings,
        project_root: Path = PROJECT_ROOT,
        codec: TokenCodec | None = None,
    ) -> None:
        self.settings = settings
        self.path = project_root / settings.documents_path
        self._codec = codec
        self._codec_lock = Lock()

    def _read_records(self, connection: sqlite3.Connection) -> list[DocumentRecord]:
        exists = connection.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'documents'"
        ).fetchone()
        if not exists:
            return []
        return [
            DocumentRecord.model_validate_json(row[0])
            for row in connection.execute(
                "SELECT record_json FROM documents ORDER BY created_at DESC, id DESC"
            )
        ]

    def _connect_readonly(self) -> sqlite3.Connection:
        return sqlite3.connect(self.path.resolve().as_uri() + "?mode=ro", uri=True, timeout=30)

    def list_documents(self) -> list[DocumentRecord]:
        """Return newest-first document metadata without loading a tokenizer or creating a DB."""

        if not self.path.exists():
            return []
        try:
            with closing(self._connect_readonly()) as connection:
                return self._read_records(connection)
        except (OSError, sqlite3.Error, ValidationError) as error:
            raise DocumentUploadError(
                "storage_unavailable", "Document storage is temporarily unavailable.", 503
            ) from error

    def snapshot(self) -> tuple[str, list[DocumentChunk]]:
        """Read an internally consistent revision and its chunks in one read transaction."""

        fingerprint = hashlib.sha256()
        if not self.path.exists():
            return fingerprint.hexdigest(), []
        try:
            with closing(self._connect_readonly()) as connection:
                connection.execute("BEGIN")
                records = self._read_records(connection)
                if not records:
                    return fingerprint.hexdigest(), []
                chunks: list[DocumentChunk] = []
                for row in connection.execute(
                    "SELECT payload FROM chunks ORDER BY document_id, chunk_index"
                ):
                    fingerprint.update(row[0].encode("utf-8"))
                    fingerprint.update(b"\n")
                    chunks.append(DocumentChunk.model_validate_json(row[0]))
                return fingerprint.hexdigest(), chunks
        except (OSError, sqlite3.Error, ValidationError) as error:
            raise DocumentUploadError(
                "storage_unavailable", "Document storage is temporarily unavailable.", 503
            ) from error

    def _prepare_chunks(self, document: RawDocument) -> list[DocumentChunk]:
        try:
            with self._codec_lock:
                if self._codec is None:
                    self._codec = HuggingFaceTokenCodec.from_pretrained(
                        self.settings.model_name, self.settings.model_revision
                    )
                chunker_class = (
                    FixedTokenChunker
                    if self.settings.chunk_strategy == "fixed"
                    else RecursiveChunker
                )
                chunker = chunker_class(
                    self._codec,
                    self.settings.chunk_size_tokens,
                    self.settings.chunk_overlap_tokens,
                )
                return build_chunks([document], chunker)
        except Exception as error:
            raise DocumentUploadError(
                "processing_failed", "The document could not be processed. Please retry.", 503
            ) from error

    def upload(self, filename: str, data: bytes) -> UploadResult:
        """Validate and chunk before atomically inserting, deduplicating, and checking quotas."""

        safe_filename = _safe_filename(filename)
        if len(data) > self.settings.upload_max_bytes:
            raise DocumentUploadError(
                "file_too_large",
                f"Documents must be at most {self.settings.upload_max_bytes} bytes.",
                413,
            )
        text = _decode_document(data)
        digest = hashlib.sha256(data).hexdigest()
        # Identical content remains idempotent even when the configured tokenizer is unavailable.
        existing = next(
            (record for record in self.list_documents() if record.sha256 == digest), None
        )
        if existing is not None:
            return UploadResult(document=existing, duplicate=True)
        document_id = f"upload_{digest}"
        document = RawDocument(
            document_id=document_id,
            source=f"uploads/{safe_filename}",
            title=_title(safe_filename, text),
            text=text,
            metadata={
                "filename": safe_filename,
                "file_type": Path(safe_filename).suffix.removeprefix("."),
                "sha256": digest,
                "chunk_size_tokens": self.settings.chunk_size_tokens,
                "overlap_tokens": self.settings.chunk_overlap_tokens,
            },
        )
        chunks = self._prepare_chunks(document)
        if not chunks:
            raise DocumentUploadError("empty_document", "The document has no text to index.", 422)
        if len(chunks) > self.settings.upload_max_chunks:
            raise DocumentUploadError(
                "chunk_limit_exceeded", "The document exceeds the library's chunk capacity.", 409
            )
        record = DocumentRecord(
            id=document_id,
            filename=safe_filename,
            title=document.title,
            size_bytes=len(data),
            sha256=digest,
            created_at=datetime.now(UTC),
            chunk_count=len(chunks),
            chunk_strategy=self.settings.chunk_strategy,
            chunk_size_tokens=self.settings.chunk_size_tokens,
            overlap_tokens=self.settings.chunk_overlap_tokens,
        )
        return self._persist(record, document.text, chunks)

    def _persist(
        self, record: DocumentRecord, text: str, chunks: list[DocumentChunk]
    ) -> UploadResult:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with closing(sqlite3.connect(self.path, timeout=30)) as connection, connection:
                connection.execute("PRAGMA foreign_keys = ON")
                connection.execute("BEGIN IMMEDIATE")
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS documents ("
                    "id TEXT PRIMARY KEY, sha256 TEXT UNIQUE NOT NULL, created_at TEXT NOT NULL, "
                    "chunk_count INTEGER NOT NULL, record_json TEXT NOT NULL, content TEXT NOT NULL)"
                )
                connection.execute(
                    "CREATE TABLE IF NOT EXISTS chunks ("
                    "document_id TEXT NOT NULL REFERENCES documents(id), "
                    "chunk_index INTEGER NOT NULL, payload TEXT NOT NULL, "
                    "PRIMARY KEY (document_id, chunk_index))"
                )
                existing = connection.execute(
                    "SELECT record_json FROM documents WHERE sha256 = ?", (record.sha256,)
                ).fetchone()
                if existing:
                    return UploadResult(
                        document=DocumentRecord.model_validate_json(existing[0]), duplicate=True
                    )
                document_count, chunk_count = connection.execute(
                    "SELECT COUNT(*), COALESCE(SUM(chunk_count), 0) FROM documents"
                ).fetchone()
                if document_count >= self.settings.upload_max_documents:
                    raise DocumentUploadError(
                        "document_limit_exceeded", "The document library is full.", 409
                    )
                if chunk_count + len(chunks) > self.settings.upload_max_chunks:
                    raise DocumentUploadError(
                        "chunk_limit_exceeded",
                        "The document library's chunk capacity is full.",
                        409,
                    )
                connection.execute(
                    "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?)",
                    (
                        record.id,
                        record.sha256,
                        record.created_at.isoformat(),
                        record.chunk_count,
                        record.model_dump_json(),
                        text,
                    ),
                )
                connection.executemany(
                    "INSERT INTO chunks (document_id, chunk_index, payload) VALUES (?, ?, ?)",
                    [(record.id, chunk.chunk_index, chunk.model_dump_json()) for chunk in chunks],
                )
            return UploadResult(document=record, duplicate=False)
        except (OSError, sqlite3.Error, ValidationError) as error:
            raise DocumentUploadError(
                "storage_unavailable", "Document storage is temporarily unavailable.", 503
            ) from error
