"""Public records returned by the document library and upload API."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field


class DocumentRecord(BaseModel):
    """A fully ingested document; incomplete uploads are never visible."""

    model_config = ConfigDict(frozen=True)

    id: str
    filename: str
    title: str
    size_bytes: int = Field(gt=0)
    sha256: str
    created_at: datetime
    chunk_count: int = Field(gt=0)
    chunk_strategy: Literal["fixed", "recursive"]
    chunk_size_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)
    status: Literal["ready"] = "ready"


class UploadResult(BaseModel):
    """A newly stored document or the record for identical existing bytes."""

    model_config = ConfigDict(frozen=True)

    document: DocumentRecord
    duplicate: bool
