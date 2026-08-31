"""Retrieval contracts shared by every search strategy."""

from dataclasses import dataclass, field
from typing import Protocol

from backend.app.ingestion.models import DocumentChunk


@dataclass(frozen=True, slots=True)
class RetrievalResult:
    """A ranked passage with transparent component scores."""

    chunk: DocumentChunk
    score: float
    rank: int
    retriever: str
    component_scores: dict[str, float] = field(default_factory=dict)


class Retriever(Protocol):
    """Minimal search interface used by evaluation and RAG."""

    name: str

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return up to ``top_k`` ranked chunks."""
