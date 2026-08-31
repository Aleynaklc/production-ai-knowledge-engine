"""Versionable JSONL contracts for retrieval relevance judgments."""

import json
import re
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.ingestion.models import DocumentChunk

QueryCategory = Literal["direct", "semantic", "adversarial", "multi_hop", "unanswerable"]


class SourceEvaluationQuery(BaseModel):
    """Human-reviewable query labels that refer to stable source text."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    category: QueryCategory
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_texts: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_labels(self) -> "SourceEvaluationQuery":
        if self.category == "unanswerable":
            if self.relevant_document_ids or self.relevant_texts:
                raise ValueError("Unanswerable queries cannot have relevance labels")
        elif not self.relevant_document_ids or not self.relevant_texts:
            raise ValueError("Answerable queries require document IDs and source anchors")
        return self


class RetrievalEvaluationQuery(BaseModel):
    """Resolved labels tied to a particular chunking run."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    category: QueryCategory
    relevant_document_ids: list[str] = Field(default_factory=list)
    relevant_chunk_ids: list[str] = Field(default_factory=list)


def _normalized_tokens(text: str) -> set[str]:
    return set(re.findall(r"[a-z0-9]+", text.casefold()))


def _resolve_anchor(anchor: str, candidates: list[DocumentChunk]) -> DocumentChunk:
    normalized_anchor = " ".join(anchor.casefold().split())
    for chunk in candidates:
        if normalized_anchor in " ".join(chunk.text.casefold().split()):
            return chunk

    anchor_tokens = _normalized_tokens(anchor)
    scored = [
        (len(anchor_tokens & _normalized_tokens(chunk.text)) / max(1, len(anchor_tokens)), chunk)
        for chunk in candidates
    ]
    score, best = max(scored, key=lambda item: item[0], default=(0.0, None))
    if best is None or score < 0.6:
        raise ValueError(f"Could not resolve relevance anchor: {anchor!r}")
    return best


def resolve_queries(
    queries: list[SourceEvaluationQuery], chunks: list[DocumentChunk]
) -> list[RetrievalEvaluationQuery]:
    """Resolve stable source anchors to deterministic IDs for one chunking strategy."""

    by_document: dict[str, list[DocumentChunk]] = {}
    for chunk in chunks:
        by_document.setdefault(chunk.document_id, []).append(chunk)

    resolved: list[RetrievalEvaluationQuery] = []
    for query in queries:
        missing = set(query.relevant_document_ids) - by_document.keys()
        if missing:
            raise ValueError(f"Unknown document IDs for {query.id}: {sorted(missing)}")
        candidates = [
            chunk
            for document_id in query.relevant_document_ids
            for chunk in by_document[document_id]
        ]
        chunk_ids = sorted(
            {_resolve_anchor(anchor, candidates).chunk_id for anchor in query.relevant_texts}
        )
        resolved.append(
            RetrievalEvaluationQuery(
                id=query.id,
                question=query.question,
                category=query.category,
                relevant_document_ids=query.relevant_document_ids,
                relevant_chunk_ids=chunk_ids,
            )
        )
    return resolved


def read_queries[QueryModel: BaseModel](path: Path, model: type[QueryModel]) -> list[QueryModel]:
    """Read and validate one JSON object per line."""

    return [
        model.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def write_queries(path: Path, queries: list[RetrievalEvaluationQuery]) -> None:
    """Persist resolved labels in a stable order."""

    path.parent.mkdir(parents=True, exist_ok=True)
    records = [json.dumps(query.model_dump(mode="json"), ensure_ascii=False) for query in queries]
    path.write_text("\n".join(records) + "\n", encoding="utf-8")
