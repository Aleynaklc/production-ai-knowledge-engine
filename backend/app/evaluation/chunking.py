"""Isolated, source-labeled comparisons of chunking strategies and token windows."""

import hashlib
import json
from collections.abc import Callable
from time import perf_counter
from typing import Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator
from qdrant_client import QdrantClient

from backend.app.embeddings.base import Embedder
from backend.app.evaluation.dataset import (
    RetrievalEvaluationQuery,
    SourceEvaluationQuery,
    resolve_queries,
)
from backend.app.evaluation.metrics import RetrievalEvaluation, evaluate_retriever
from backend.app.ingestion.chunkers import FixedTokenChunker, RecursiveChunker, TokenCodec
from backend.app.ingestion.models import RawDocument
from backend.app.ingestion.pipeline import build_chunks
from backend.app.retrieval.dense import QdrantDenseRetriever


class ChunkingVariant(BaseModel):
    """One valid, uniquely named point in a chunking experiment."""

    model_config = ConfigDict(frozen=True)

    strategy: Literal["fixed", "recursive"]
    chunk_size_tokens: int = Field(gt=0)
    overlap_tokens: int = Field(ge=0)

    @model_validator(mode="after")
    def validate_overlap(self) -> "ChunkingVariant":
        if self.overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("overlap_tokens must be smaller than chunk_size_tokens")
        return self

    @property
    def name(self) -> str:
        return f"{self.strategy}_{self.chunk_size_tokens}_{self.overlap_tokens}"


class ChunkingBenchmarkConfig(BaseModel):
    """Retrieval and timing controls held constant across all variants."""

    model_config = ConfigDict(frozen=True)

    top_k: int = Field(default=5, gt=0)
    repetitions: int = Field(default=3, gt=0)
    warmup_runs: int = Field(default=1, ge=0)


class ChunkingVariantResult(BaseModel):
    """Chunk statistics, separate build costs, and every retrieval trial."""

    model_config = ConfigDict(frozen=True)

    variant: ChunkingVariant
    chunk_count: int
    chunk_tokens_total: int
    chunk_tokens_min: int
    chunk_tokens_mean: float
    chunk_tokens_p50: float
    chunk_tokens_p95: float
    chunk_tokens_max: int
    token_expansion_ratio: float
    chunk_text_utf8_bytes: int
    raw_float32_vector_bytes: int
    embedding_truncated_chunk_count: int | None
    chunking_ms: float
    embedding_documents_ms: float
    index_build_ms: float
    index_non_embedding_ms: float
    ingestion_ms: float
    query_latency_mean_ms: float
    query_latency_p50_ms: float
    query_latency_p95_ms: float
    resolved_queries: list[RetrievalEvaluationQuery]
    trials: list[RetrievalEvaluation]


class ChunkingBenchmarkResult(BaseModel):
    """Deterministic input fingerprints and measured per-variant results."""

    model_config = ConfigDict(frozen=True)

    configuration: ChunkingBenchmarkConfig
    document_count: int
    source_token_count: int
    source_sha256: str
    labels_sha256: str
    variants: list[ChunkingVariantResult]


class _TimedEmbedder:
    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.documents_ms = 0.0

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = perf_counter()
        result = self.embedder.embed_documents(texts)
        self.documents_ms += (perf_counter() - started) * 1_000
        return result

    def embed_query(self, text: str) -> list[float]:
        return self.embedder.embed_query(text)


def _fingerprint(records: list[dict[str, object]]) -> str:
    payload = json.dumps(records, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def benchmark_chunking(
    documents: list[RawDocument],
    queries: list[SourceEvaluationQuery],
    codec: TokenCodec,
    embedder: Embedder,
    variants: list[ChunkingVariant],
    *,
    config: ChunkingBenchmarkConfig | None = None,
    embedding_token_count: Callable[[str], int] | None = None,
    embedding_max_tokens: int | None = None,
) -> ChunkingBenchmarkResult:
    """Rechunk and rebuild an independent in-memory dense index for every variant.

    Ingestion time covers chunking (including token metadata) and embedding/indexing;
    shared file loading and model initialization are reported separately by the CLI.
    The same embedder instance is used throughout and source anchors are re-resolved.
    """

    config = config or ChunkingBenchmarkConfig()
    if not documents or not any(document.text.strip() for document in documents):
        raise ValueError("Chunking benchmark requires non-empty documents")
    if len({document.document_id for document in documents}) != len(documents):
        raise ValueError("Document IDs must be unique")
    if not queries or not any(query.category != "unanswerable" for query in queries):
        raise ValueError("Chunking benchmark requires at least one answerable query")
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("Query IDs must be unique")
    if not variants or len({variant.name for variant in variants}) != len(variants):
        raise ValueError("Supply at least one variant and no duplicate variants")
    if (embedding_token_count is None) != (embedding_max_tokens is None):
        raise ValueError("Embedding token counter and maximum length must be supplied together")
    if embedding_max_tokens is not None and embedding_max_tokens <= 0:
        raise ValueError("Embedding maximum length must be positive")
    if embedder.dimension <= 0:
        raise ValueError("Embedding dimension must be positive")

    source_tokens = sum(codec.count(document.text) for document in documents)
    cutoffs = tuple(sorted({1, config.top_k, *[k for k in (3, 5) if k <= config.top_k]}))
    results: list[ChunkingVariantResult] = []
    for variant in variants:
        chunker_type = FixedTokenChunker if variant.strategy == "fixed" else RecursiveChunker
        chunker = chunker_type(codec, variant.chunk_size_tokens, variant.overlap_tokens)
        started = perf_counter()
        chunks = build_chunks(documents, chunker)
        chunking_ms = (perf_counter() - started) * 1_000
        if not chunks:
            raise ValueError(f"Variant {variant.name} produced no chunks")
        try:
            resolved = resolve_queries(queries, chunks)
        except ValueError as exc:
            raise ValueError(f"Variant {variant.name}: {exc}") from exc
        token_counts = [codec.count(chunk.text) for chunk in chunks]
        truncated = (
            sum(embedding_token_count(chunk.text) > embedding_max_tokens for chunk in chunks)
            if embedding_token_count is not None and embedding_max_tokens is not None
            else None
        )
        timed_embedder = _TimedEmbedder(embedder)
        # Explicit in-memory clients cannot lock, recreate, or overwrite application indexes.
        client = QdrantClient(location=":memory:")
        try:
            retriever = QdrantDenseRetriever(timed_embedder, "chunking_benchmark", client=client)
            started = perf_counter()
            retriever.index(chunks)
            index_ms = (perf_counter() - started) * 1_000
            for _ in range(config.warmup_runs):
                retriever.retrieve(queries[0].question, top_k=config.top_k)
            trials = [
                evaluate_retriever(retriever, resolved, max_k=config.top_k, cutoffs=cutoffs)
                for _ in range(config.repetitions)
            ]
        finally:
            client.close()
        latencies = [item.latency_ms for trial in trials for item in trial.per_query]
        results.append(
            ChunkingVariantResult(
                variant=variant,
                chunk_count=len(chunks),
                chunk_tokens_total=sum(token_counts),
                chunk_tokens_min=min(token_counts),
                chunk_tokens_mean=round(float(np.mean(token_counts)), 6),
                chunk_tokens_p50=round(float(np.percentile(token_counts, 50)), 6),
                chunk_tokens_p95=round(float(np.percentile(token_counts, 95)), 6),
                chunk_tokens_max=max(token_counts),
                token_expansion_ratio=round(sum(token_counts) / max(1, source_tokens), 6),
                chunk_text_utf8_bytes=sum(len(chunk.text.encode("utf-8")) for chunk in chunks),
                raw_float32_vector_bytes=len(chunks) * embedder.dimension * 4,
                embedding_truncated_chunk_count=truncated,
                chunking_ms=round(chunking_ms, 6),
                embedding_documents_ms=round(timed_embedder.documents_ms, 6),
                index_build_ms=round(index_ms, 6),
                index_non_embedding_ms=round(max(0.0, index_ms - timed_embedder.documents_ms), 6),
                ingestion_ms=round(chunking_ms + index_ms, 6),
                query_latency_mean_ms=round(float(np.mean(latencies)), 6),
                query_latency_p50_ms=round(float(np.percentile(latencies, 50)), 6),
                query_latency_p95_ms=round(float(np.percentile(latencies, 95)), 6),
                resolved_queries=resolved,
                trials=trials,
            )
        )

    return ChunkingBenchmarkResult(
        configuration=config,
        document_count=len(documents),
        source_token_count=source_tokens,
        source_sha256=_fingerprint([document.model_dump(mode="json") for document in documents]),
        labels_sha256=_fingerprint([query.model_dump(mode="json") for query in queries]),
        variants=results,
    )


def chunking_markdown(result: ChunkingBenchmarkResult) -> str:
    """Render ranking quality together with ingestion and query cost."""

    k = str(result.configuration.top_k)
    rows = [
        f"| Variant | Chunks | Mean tokens | Recall@{k} | Hit rate@{k} | MRR@{k} | "
        "Ingestion ms | Index ms | Query p50/p95 ms |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for item in result.variants:
        recall = float(np.mean([trial.metrics.recall_at_k[k] for trial in item.trials]))
        mrr = float(np.mean([trial.metrics.mrr_at_k[k] for trial in item.trials]))
        hit_rate = float(
            np.mean(
                [
                    np.mean(
                        [
                            record.first_relevant_rank is not None
                            for record in trial.per_query
                            if record.relevant_chunk_ids
                        ]
                    )
                    for trial in item.trials
                ]
            )
        )
        rows.append(
            f"| {item.variant.name} | {item.chunk_count} | {item.chunk_tokens_mean:.1f} | "
            f"{recall:.3f} | {hit_rate:.3f} | {mrr:.3f} | {item.ingestion_ms:.2f} | "
            f"{item.index_build_ms:.2f} | "
            f"{item.query_latency_p50_ms:.2f} / {item.query_latency_p95_ms:.2f} |"
        )
    return "\n".join(rows)
