"""Bounded PDF/DOCX/text extraction with honest source locations."""

import base64
import json
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from zipfile import ZipFile

from docx import Document
from docx.table import Table
from docx.text.paragraph import Paragraph
from pydantic import BaseModel
from pypdf import PdfReader

from backend.app.config import Settings
from backend.app.documents.validation import DocumentUploadError, _decode_document


class SourceUnit(BaseModel):
    number: int
    kind: str
    text: str


def extract(filename: str, data: bytes, settings: Settings) -> list[SourceUnit]:
    """Keep untrusted binary parsing out of the API process, with a hard timeout."""
    if Path(filename).suffix.lower() not in {".pdf", ".docx"}:
        return extract_in_process(filename, data, settings)
    payload = {
        "filename": filename,
        "data": base64.b64encode(data).decode(),
        "max_bytes": settings.upload_max_expanded_bytes,
        "max_pages": settings.upload_max_pages,
    }
    try:
        result = subprocess.run(
            [sys.executable, "-m", "backend.app.workspaces.parser_worker"],
            input=json.dumps(payload),
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
        if result.returncode not in {0, 1} or not result.stdout:
            raise ValueError("Parser stopped")
        parsed = json.loads(result.stdout)
        if "error" in parsed:
            raise DocumentUploadError(parsed["error"]["code"], parsed["error"]["message"], 422)
        return [SourceUnit.model_validate(item) for item in parsed["units"]]
    except (subprocess.TimeoutExpired, OSError, ValueError) as error:
        if isinstance(error, DocumentUploadError):
            raise
        raise DocumentUploadError(
            "processing_limit", "Document exceeded parser limits or could not be read.", 422
        ) from error


def extract_in_process(filename: str, data: bytes, settings: Settings) -> list[SourceUnit]:
    """Preserve PDF pages; DOCX paragraphs/tables are sections, never invented pages."""
    suffix = Path(filename).suffix.lower()
    units: list[SourceUnit] = []
    try:
        if suffix == ".pdf":
            reader = PdfReader(BytesIO(data))
            if reader.is_encrypted:
                raise ValueError("Encrypted PDF")
            if len(reader.pages) > settings.upload_max_pages:
                raise ValueError("Too many pages")
            for number, page in enumerate(reader.pages, 1):
                contents = page.get_contents()
                if contents and len(contents.get_data()) > settings.upload_max_expanded_bytes:
                    raise ValueError("Expanded page exceeds limit")
                units.append(SourceUnit(number=number, kind="page", text=page.extract_text() or ""))
        elif suffix == ".docx":
            with ZipFile(BytesIO(data)) as archive:
                if (
                    sum(item.file_size for item in archive.infolist())
                    > settings.upload_max_expanded_bytes
                ):
                    raise ValueError("Expanded DOCX exceeds limit")
                if len(archive.infolist()) > 10_000:
                    raise ValueError("Too many DOCX entries")
            document = Document(BytesIO(data))
            for block in document.iter_inner_content():
                text = block.text if isinstance(block, Paragraph) else ""
                if isinstance(block, Table):
                    text = "\n".join(
                        " | ".join(cell.text for cell in row.cells) for row in block.rows
                    )
                if text.strip():
                    units.append(SourceUnit(number=len(units) + 1, kind="section", text=text))
        else:
            units = [SourceUnit(number=1, kind="section", text=_decode_document(data))]
        if (
            sum(len(unit.text.encode("utf-8")) for unit in units)
            > settings.upload_max_expanded_bytes
        ):
            raise ValueError("Extracted text exceeds limit")
        if not any(unit.text.strip() for unit in units):
            raise DocumentUploadError(
                "no_extractable_text",
                "No readable text found. Scanned PDFs need OCR before upload.",
                422,
            )
        return units
    except DocumentUploadError:
        raise
    except Exception as error:
        raise DocumentUploadError(
            "invalid_document", "Document is damaged, encrypted, or exceeds processing limits.", 422
        ) from error
