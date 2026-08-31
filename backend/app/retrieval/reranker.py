"""Cross-encoder scoring for a retrieved candidate set."""

from typing import Protocol, cast

import numpy as np
from sentence_transformers import CrossEncoder

from backend.app.llm.model import DeviceRequest, resolve_device
from backend.app.retrieval.models import RetrievalResult, Retriever


class PairScorer(Protocol):
    """Assign one relevance score to every query-passage pair."""

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Return scores in passage order."""


class CrossEncoderScorer:
    """Sentence Transformers cross-encoder adapter."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        requested_device: DeviceRequest = "auto",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = resolve_device(requested_device)
        self._model = CrossEncoder(
            model_name,
            revision=revision,
            device=self.device.type,
        )

    def score(self, query: str, passages: list[str]) -> list[float]:
        """Score query-passage pairs as a batch."""

        pairs = [(query, passage) for passage in passages]
        values = self._model.predict(pairs, show_progress_bar=False)
        return cast(list[float], np.asarray(values, dtype=float).reshape(-1).tolist())


class RerankedRetriever:
    """Rerank a broader first-stage candidate list and return the best passages."""

    name = "hybrid_reranked"

    def __init__(self, base: Retriever, scorer: PairScorer, candidate_k: int = 20) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self.base = base
        self.scorer = scorer
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Retrieve candidates and replace their order with cross-encoder scores."""

        candidates = self.base.retrieve(query, self.candidate_k)
        scores = self.scorer.score(query, [result.chunk.text for result in candidates])
        if len(scores) != len(candidates):
            raise ValueError("Reranker returned a different number of scores than candidates")
        order = sorted(range(len(candidates)), key=lambda index: (-scores[index], index))[:top_k]
        results: list[RetrievalResult] = []
        for rank, index in enumerate(order, start=1):
            candidate = candidates[index]
            component_scores = {**candidate.component_scores, "reranker": scores[index]}
            results.append(
                RetrievalResult(
                    chunk=candidate.chunk,
                    score=scores[index],
                    rank=rank,
                    retriever=self.name,
                    component_scores=component_scores,
                )
            )
        return results
