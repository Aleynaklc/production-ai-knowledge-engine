"""Deterministic BM25 lexical retrieval."""

import re

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from backend.app.ingestion.models import DocumentChunk
from backend.app.retrieval.models import RetrievalResult

TOKEN_PATTERN = re.compile(r"[a-z0-9]+")


def tokenize_for_bm25(text: str) -> list[str]:
    """Apply a small, inspectable tokenizer suited to the English demo corpus."""

    return TOKEN_PATTERN.findall(text.casefold())


class BM25Retriever:
    """Rank the in-memory corpus using Okapi BM25."""

    name = "bm25"

    def __init__(self, chunks: list[DocumentChunk]) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        self.chunks = chunks
        corpus = [tokenize_for_bm25(chunk.text) for chunk in chunks]
        # Valid uploads can contain only symbols or non-Latin text. In that case
        # this English tokenizer has no lexical evidence; dense retrieval still works.
        self._index = BM25Okapi(corpus) if any(corpus) else None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return highest-scoring chunks with stable tie-breaking."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self._index is None:
            return []
        scores = np.asarray(self._index.get_scores(tokenize_for_bm25(query)), dtype=float)
        order = sorted(range(len(self.chunks)), key=lambda index: (-scores[index], index))
        results: list[RetrievalResult] = []
        for rank, index in enumerate(order[:top_k], start=1):
            score = float(scores[index])
            results.append(
                RetrievalResult(
                    chunk=self.chunks[index],
                    score=score,
                    rank=rank,
                    retriever=self.name,
                    component_scores={self.name: score},
                )
            )
        return results
