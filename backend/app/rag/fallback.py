"""High-confidence extractive fallback for weak generative-model outputs."""

import re

from backend.app.rag.context import ContextBundle
from backend.app.retrieval.reranker import PairScorer

SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+|\n{2,}")


class ExtractiveFallback:
    """Select one verbatim source sentence when its cross-encoder score is strong."""

    def __init__(self, scorer: PairScorer, minimum_score: float = 1.0) -> None:
        self.scorer = scorer
        self.minimum_score = minimum_score

    def answer(self, query: str, context: ContextBundle) -> str | None:
        """Return a cited source sentence or ``None`` when support is too weak."""

        candidates: list[tuple[str, str]] = []
        for source in context.sources:
            for sentence in SENTENCE_SPLIT.split(source.text):
                normalized = sentence.strip()
                if not normalized or normalized.startswith("#") or len(normalized) < 12:
                    continue
                candidates.append((source.citation_id, normalized))
        if not candidates:
            return None
        scores = self.scorer.score(query, [sentence for _, sentence in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Fallback scorer returned an unexpected number of scores")
        best_index = max(range(len(candidates)), key=lambda index: scores[index])
        if scores[best_index] < self.minimum_score:
            return None
        citation_id, sentence = candidates[best_index]
        return f"[{citation_id}] {sentence}"
