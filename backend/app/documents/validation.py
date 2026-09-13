"""Lightweight document validation shared with isolated parser processes."""

import re
import unicodedata
from pathlib import Path

from backend.app.ingestion.cleaner import clean_text


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
