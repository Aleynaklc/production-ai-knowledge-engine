"""Deterministic BM25 lexical retrieval."""

import re
import unicodedata

import numpy as np
from rank_bm25 import BM25Okapi  # type: ignore[import-untyped]

from backend.app.ingestion.models import DocumentChunk
from backend.app.retrieval.models import RetrievalResult

TOKEN_PATTERN = re.compile(r"[^\W_]+", re.UNICODE)


def lexical_form(token: str) -> str:
    """Conservative English plural matching; never change the stored source text."""
    if len(token) > 4 and token.endswith("ies"):
        return token[:-3] + "y"
    if len(token) > 4 and token.endswith("s") and not token.endswith(("ss", "us", "is")):
        return token[:-1]
    return token


def tokenize_for_bm25(text: str) -> list[str]:
    """Keep non-English words and normalize accents without dropping their letters."""
    text = unicodedata.normalize("NFKD", text.casefold().replace("ı", "i"))
    text = "".join(char for char in text if not unicodedata.combining(char))
    return [lexical_form(token) for token in TOKEN_PATTERN.findall(text)]


class BM25Retriever:
    """Rank the in-memory corpus using Okapi BM25."""

    name = "bm25"

    def __init__(self, chunks: list[DocumentChunk], *, include_metadata: bool = False) -> None:
        if not chunks:
            raise ValueError("BM25 requires at least one chunk")
        self.chunks = chunks
        corpus = [
            tokenize_for_bm25(
                f"{chunk.source} {chunk.metadata.get('section_heading', '')}\n{chunk.text}"
                if include_metadata
                else chunk.text
            )
            for chunk in chunks
        ]
        self._terms = [set(tokens) for tokens in corpus]
        # Symbol-only uploads have no lexical evidence; dense retrieval still works.
        self._index = BM25Okapi(corpus) if any(corpus) else None

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Return highest-scoring chunks with stable tie-breaking."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        if self._index is None:
            return []
        query_tokens = tokenize_for_bm25(query)
        scores = np.asarray(self._index.get_scores(query_tokens), dtype=float)
        # A zero/negative BM25 score can still be a real match in tiny corpora.
        # Filter by shared terms, not score, so unrelated chunks never gain RRF votes.
        matched = [
            index for index, terms in enumerate(self._terms) if terms.intersection(query_tokens)
        ]
        order = sorted(matched, key=lambda index: (-scores[index], index))
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
