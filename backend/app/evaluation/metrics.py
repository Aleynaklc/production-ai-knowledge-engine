"""Offline ranking metrics with per-query latency and diagnostics."""

import math
from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from backend.app.evaluation.dataset import RetrievalEvaluationQuery
from backend.app.retrieval.models import Retriever


class QueryEvaluation(BaseModel):
    """Ranked IDs and metrics for one labeled question."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    category: str
    relevant_chunk_ids: list[str]
    retrieved_chunk_ids: list[str]
    first_relevant_rank: int | None
    latency_ms: float


class RetrievalMetrics(BaseModel):
    """Aggregate metrics over answerable queries."""

    model_config = ConfigDict(frozen=True)

    query_count: int
    answerable_query_count: int
    unanswerable_query_count: int
    recall_at_k: dict[str, float]
    hit_rate_at_k: dict[str, float] = Field(default_factory=dict)
    precision_at_k: dict[str, float]
    mrr_at_k: dict[str, float]
    ndcg_at_k: dict[str, float]
    mean_latency_ms: float
    p95_latency_ms: float
    p50_latency_ms: float = 0.0


class RetrievalEvaluation(BaseModel):
    """Complete result for one retriever configuration."""

    model_config = ConfigDict(frozen=True)

    retriever: str
    metrics: RetrievalMetrics
    per_query: list[QueryEvaluation]


def _first_relevant_rank(retrieved: list[str], relevant: set[str], top_k: int) -> int | None:
    for rank, chunk_id in enumerate(retrieved[:top_k], start=1):
        if chunk_id in relevant:
            return rank
    return None


def _ndcg(retrieved: list[str], relevant: set[str], top_k: int) -> float:
    seen: set[str] = set()
    gains: list[float] = []
    for rank, chunk_id in enumerate(retrieved[:top_k], start=1):
        if chunk_id in relevant and chunk_id not in seen:
            gains.append(1.0 / math.log2(rank + 1))
        seen.add(chunk_id)
    dcg = sum(gains)
    ideal_count = min(len(relevant), top_k)
    ideal = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_count + 1))
    return dcg / ideal if ideal else 0.0


def evaluate_retriever(
    retriever: Retriever,
    queries: list[RetrievalEvaluationQuery],
    *,
    max_k: int = 5,
    cutoffs: tuple[int, ...] = (1, 3, 5),
) -> RetrievalEvaluation:
    """Run a retriever once per query and calculate Recall, Precision, MRR, and NDCG."""

    if not queries:
        raise ValueError("Evaluation requires at least one query")
    if max_k <= 0 or not cutoffs or any(cutoff <= 0 for cutoff in cutoffs):
        raise ValueError("max_k and metric cutoffs must be positive")
    if len(cutoffs) != len(set(cutoffs)):
        raise ValueError("Metric cutoffs must be unique")
    if max(cutoffs) > max_k:
        raise ValueError("Every metric cutoff must be at most max_k")
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("Evaluation query IDs must be unique")
    if not any(query.relevant_chunk_ids for query in queries):
        raise ValueError("Evaluation requires at least one answerable query")

    per_query: list[QueryEvaluation] = []
    for query in queries:
        started_at = perf_counter()
        results = retriever.retrieve(query.question, top_k=max_k)
        elapsed_ms = (perf_counter() - started_at) * 1_000
        # Duplicate results occupy a rank but cannot earn credit more than once.
        retrieved = [result.chunk.chunk_id for result in results[:max_k]]
        relevant = set(query.relevant_chunk_ids)
        per_query.append(
            QueryEvaluation(
                query_id=query.id,
                category=query.category,
                relevant_chunk_ids=query.relevant_chunk_ids,
                retrieved_chunk_ids=retrieved,
                first_relevant_rank=_first_relevant_rank(retrieved, relevant, max_k),
                latency_ms=round(elapsed_ms, 6),
            )
        )

    answerable = [item for item in per_query if item.relevant_chunk_ids]
    if not answerable:
        raise ValueError("Evaluation requires at least one answerable query")

    recall: dict[str, float] = {}
    hit_rate: dict[str, float] = {}
    precision: dict[str, float] = {}
    mrr: dict[str, float] = {}
    ndcg: dict[str, float] = {}
    for cutoff in cutoffs:
        recalls: list[float] = []
        hit_rates: list[float] = []
        precisions: list[float] = []
        reciprocal_ranks: list[float] = []
        ndcgs: list[float] = []
        for item in answerable:
            relevant = set(item.relevant_chunk_ids)
            retrieved = item.retrieved_chunk_ids[:cutoff]
            hits = len(set(retrieved) & relevant)
            rank = _first_relevant_rank(retrieved, relevant, cutoff)
            recalls.append(hits / len(relevant))
            hit_rates.append(float(hits > 0))
            precisions.append(hits / cutoff)
            reciprocal_ranks.append(1.0 / rank if rank is not None else 0.0)
            ndcgs.append(_ndcg(retrieved, relevant, cutoff))
        key = str(cutoff)
        recall[key] = round(float(np.mean(recalls)), 6)
        hit_rate[key] = round(float(np.mean(hit_rates)), 6)
        precision[key] = round(float(np.mean(precisions)), 6)
        mrr[key] = round(float(np.mean(reciprocal_ranks)), 6)
        ndcg[key] = round(float(np.mean(ndcgs)), 6)

    latencies = [item.latency_ms for item in per_query]
    return RetrievalEvaluation(
        retriever=retriever.name,
        metrics=RetrievalMetrics(
            query_count=len(queries),
            answerable_query_count=len(answerable),
            unanswerable_query_count=len(queries) - len(answerable),
            recall_at_k=recall,
            hit_rate_at_k=hit_rate,
            precision_at_k=precision,
            mrr_at_k=mrr,
            ndcg_at_k=ndcg,
            mean_latency_ms=round(float(np.mean(latencies)), 6),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 6),
            p50_latency_ms=round(float(np.percentile(latencies, 50)), 6),
        ),
        per_query=per_query,
    )
