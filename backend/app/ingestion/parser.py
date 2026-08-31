"""Route supported source files into normalized document records."""

import re
from pathlib import Path

from backend.app.ingestion.cleaner import clean_text
from backend.app.ingestion.loaders.text import load_text_file
from backend.app.ingestion.models import RawDocument

SUPPORTED_SUFFIXES = {".md", ".txt"}


def _document_id(relative_path: Path) -> str:
    slug = re.sub(r"[^a-z0-9]+", "_", relative_path.with_suffix("").as_posix().lower())
    return slug.strip("_")


def _title(path: Path, text: str) -> str:
    if path.suffix.lower() == ".md":
        heading = re.search(r"(?m)^#\s+(.+)$", text)
        if heading:
            return heading.group(1).strip()
    return path.stem.replace("_", " ").strip().title()


def parse_document(path: Path, root: Path) -> RawDocument:
    """Parse one supported source file and attach stable source metadata."""

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_SUFFIXES:
        raise ValueError(f"Unsupported document type: {suffix}")
    relative_path = path.relative_to(root)
    text = clean_text(load_text_file(path))
    if not text:
        raise ValueError(f"Document is empty after cleaning: {relative_path}")
    return RawDocument(
        document_id=_document_id(relative_path),
        source=relative_path.as_posix(),
        title=_title(path, text),
        text=text,
        metadata={"file_type": suffix.removeprefix("."), "character_count": len(text)},
    )


def load_documents(root: Path) -> list[RawDocument]:
    """Load all supported documents below a directory in deterministic order."""

    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.suffix.lower() in SUPPORTED_SUFFIXES
    )
    documents = [parse_document(path, root) for path in paths]
    document_ids = [document.document_id for document in documents]
    if len(document_ids) != len(set(document_ids)):
        raise ValueError("Document IDs collide after path normalization")
    return documents
