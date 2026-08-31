"""Typed data contracts shared by ingestion and retrieval."""

from pydantic import BaseModel, ConfigDict, Field

type MetadataValue = str | int | float | bool | None


class RawDocument(BaseModel):
    """One normalized source document before chunking."""

    model_config = ConfigDict(frozen=True)

    document_id: str
    source: str
    title: str
    text: str
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)


class DocumentChunk(BaseModel):
    """One independently retrievable passage with source lineage."""

    model_config = ConfigDict(frozen=True)

    chunk_id: str
    document_id: str
    source: str
    text: str
    chunk_index: int = Field(ge=0)
    metadata: dict[str, MetadataValue] = Field(default_factory=dict)
