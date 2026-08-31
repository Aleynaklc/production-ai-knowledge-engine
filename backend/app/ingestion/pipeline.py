"""End-to-end ingestion orchestration and JSONL persistence."""

import hashlib
import json
from pathlib import Path
from uuid import NAMESPACE_URL, uuid5

from pydantic import BaseModel

from backend.app.ingestion.chunkers import BaseChunker
from backend.app.ingestion.models import DocumentChunk, RawDocument
from backend.app.ingestion.parser import load_documents


class IngestionSummary(BaseModel):
    """Observable ingestion statistics."""

    document_count: int
    chunk_count: int
    average_chunk_tokens: float
    average_chunk_characters: float
    strategy: str
    chunk_size_tokens: int
    overlap_tokens: int


def build_chunks(documents: list[RawDocument], chunker: BaseChunker) -> list[DocumentChunk]:
    """Chunk documents and assign deterministic UUIDs with source lineage."""

    chunks: list[DocumentChunk] = []
    for document in documents:
        passages = chunker.chunk(document.text)
        for index, passage in enumerate(passages):
            content_hash = hashlib.sha256(passage.encode("utf-8")).hexdigest()
            chunk_id = str(
                uuid5(
                    NAMESPACE_URL,
                    f"pake:{document.document_id}:{chunker.name}:{index}:{content_hash}",
                )
            )
            chunks.append(
                DocumentChunk(
                    chunk_id=chunk_id,
                    document_id=document.document_id,
                    source=document.source,
                    text=passage,
                    chunk_index=index,
                    metadata={
                        **document.metadata,
                        "title": document.title,
                        "chunking_strategy": chunker.name,
                        "character_count": len(passage),
                        "token_count": chunker.codec.count(passage),
                    },
                )
            )
    return chunks


def summarize(
    chunks: list[DocumentChunk], documents: list[RawDocument], chunker: BaseChunker
) -> IngestionSummary:
    """Calculate reproducible chunking statistics."""

    token_counts = [
        value if isinstance(value := chunk.metadata.get("token_count"), int) else 0
        for chunk in chunks
    ]
    character_counts = [len(chunk.text) for chunk in chunks]
    return IngestionSummary(
        document_count=len(documents),
        chunk_count=len(chunks),
        average_chunk_tokens=sum(token_counts) / len(token_counts) if token_counts else 0.0,
        average_chunk_characters=(
            sum(character_counts) / len(character_counts) if character_counts else 0.0
        ),
        strategy=chunker.name,
        chunk_size_tokens=chunker.chunk_size,
        overlap_tokens=chunker.overlap,
    )


def write_chunks(path: Path, chunks: list[DocumentChunk]) -> None:
    """Persist chunks as deterministic, reviewable JSON Lines."""

    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(chunk.model_dump(mode="json"), ensure_ascii=False) for chunk in chunks]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def read_chunks(path: Path) -> list[DocumentChunk]:
    """Load persisted chunks from JSON Lines."""

    return [
        DocumentChunk.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def ingest_directory(
    input_directory: Path,
    output_path: Path,
    chunker: BaseChunker,
) -> tuple[list[DocumentChunk], IngestionSummary]:
    """Load, normalize, chunk, persist, and summarize one corpus."""

    documents = load_documents(input_directory)
    chunks = build_chunks(documents, chunker)
    write_chunks(output_path, chunks)
    return chunks, summarize(chunks, documents, chunker)
