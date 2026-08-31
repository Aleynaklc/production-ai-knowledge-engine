"""Rank fusion that combines incomparable retrieval score scales."""

from collections.abc import Sequence

from backend.app.retrieval.models import RetrievalResult


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[RetrievalResult]],
    *,
    top_k: int,
    rank_constant: int = 60,
) -> list[RetrievalResult]:
    """Merge ranked lists with Reciprocal Rank Fusion (RRF)."""

    if top_k < 1:
        raise ValueError("top_k must be positive")
    if rank_constant < 1:
        raise ValueError("rank_constant must be positive")

    scores: dict[str, float] = {}
    chunks: dict[str, RetrievalResult] = {}
    components: dict[str, dict[str, float]] = {}
    for ranked in ranked_lists:
        for result in ranked:
            chunk_id = result.chunk.chunk_id
            scores[chunk_id] = scores.get(chunk_id, 0.0) + 1.0 / (rank_constant + result.rank)
            chunks.setdefault(chunk_id, result)
            values = components.setdefault(chunk_id, {})
            values.update(result.component_scores)
            values[f"{result.retriever}_rrf"] = 1.0 / (rank_constant + result.rank)

    ordered = sorted(scores, key=lambda chunk_id: (-scores[chunk_id], chunk_id))[:top_k]
    return [
        RetrievalResult(
            chunk=chunks[chunk_id].chunk,
            score=scores[chunk_id],
            rank=rank,
            retriever="hybrid_rrf",
            component_scores=components[chunk_id],
        )
        for rank, chunk_id in enumerate(ordered, start=1)
    ]
