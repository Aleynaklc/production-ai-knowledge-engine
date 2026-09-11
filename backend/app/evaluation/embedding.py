"""Stage 23: controlled encoder quality, indexing cost, and latency measurements."""

from time import perf_counter

import numpy as np
from pydantic import BaseModel, ConfigDict, Field
from qdrant_client import QdrantClient

from backend.app.embeddings.base import Embedder
from backend.app.evaluation.dataset import RetrievalEvaluationQuery
from backend.app.evaluation.metrics import RetrievalEvaluation, evaluate_retriever
from backend.app.ingestion.models import DocumentChunk
from backend.app.retrieval.dense import QdrantDenseRetriever


class EmbeddingProfile(BaseModel):
    """An immutable model revision and batch-size configuration."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1)
    model_name: str = Field(min_length=1)
    revision: str = Field(pattern=r"^[0-9a-f]{40}$")
    batch_size: int = Field(default=32, gt=0)


class LatencySummary(BaseModel):
    """Latency distribution; samples remain in the report for auditing."""

    samples_ms: list[float]
    mean_ms: float
    p50_ms: float
    p95_ms: float


def summarize_latency(samples: list[float]) -> LatencySummary:
    if not samples or any(not np.isfinite(value) or value < 0 for value in samples):
        raise ValueError("Latency samples must be nonempty, finite, and nonnegative")
    return LatencySummary(
        samples_ms=[round(value, 6) for value in samples],
        mean_ms=round(float(np.mean(samples)), 6),
        p50_ms=round(float(np.percentile(samples, 50)), 6),
        p95_ms=round(float(np.percentile(samples, 95)), 6),
    )


class EmbeddingBenchmarkResult(BaseModel):
    profile: EmbeddingProfile
    model_metadata: dict[str, str | int | float]
    model_load_ms: float
    chunk_count: int
    query_count: int
    dimension: int
    raw_float32_vector_bytes: int
    repetitions: int
    warmup_runs: int
    document_embedding: LatencySummary
    index_build: LatencySummary
    query_embedding: LatencySummary
    retrieval_latency: LatencySummary
    document_embeddings_per_second: float
    retrieval_runs: list[RetrievalEvaluation]


class _MeasuredEmbedder:
    """Time the production boundary, including CPU conversion/synchronization."""

    def __init__(self, embedder: Embedder) -> None:
        self.embedder = embedder
        self.document_ms: list[float] = []
        self.query_ms: list[float] = []

    @property
    def dimension(self) -> int:
        return self.embedder.dimension

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        started = perf_counter()
        vectors = self.embedder.embed_documents(texts)
        self.document_ms.append((perf_counter() - started) * 1_000)
        return vectors

    def embed_query(self, text: str) -> list[float]:
        started = perf_counter()
        vector = self.embedder.embed_query(text)
        self.query_ms.append((perf_counter() - started) * 1_000)
        return vector


def benchmark_embedding(
    embedder: Embedder,
    profile: EmbeddingProfile,
    chunks: list[DocumentChunk],
    queries: list[RetrievalEvaluationQuery],
    *,
    repetitions: int = 3,
    warmup_runs: int = 1,
    top_k: int = 5,
    model_load_ms: float = 0.0,
    model_metadata: dict[str, str | int | float] | None = None,
) -> EmbeddingBenchmarkResult:
    """Hold corpus/labels/search constant; never touch the serving Qdrant database."""

    if not chunks or not queries:
        raise ValueError("Embedding benchmark requires chunks and queries")
    if repetitions <= 0 or warmup_runs < 0 or top_k <= 0:
        raise ValueError("repetitions/top_k must be positive and warmup_runs nonnegative")
    if len({chunk.chunk_id for chunk in chunks}) != len(chunks):
        raise ValueError("Chunk IDs must be unique")
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("Query IDs must be unique")
    known_ids = {chunk.chunk_id for chunk in chunks}
    if any(set(query.relevant_chunk_ids) - known_ids for query in queries):
        raise ValueError("All relevance labels must refer to benchmark chunks")
    if not any(query.relevant_chunk_ids for query in queries):
        raise ValueError("Embedding benchmark requires answerable queries")
    if embedder.dimension <= 0 or not np.isfinite(model_load_ms) or model_load_ms < 0:
        raise ValueError("Embedding dimension and model-load measurement are invalid")

    timed = _MeasuredEmbedder(embedder)
    texts = [chunk.text for chunk in chunks]
    cutoffs = tuple(sorted({cutoff for cutoff in (1, 3, 5, top_k) if cutoff <= top_k}))
    for _ in range(warmup_runs):
        embedder.embed_documents(texts)
    index_ms: list[float] = []
    runs: list[RetrievalEvaluation] = []
    client = QdrantClient(":memory:")
    try:
        retriever = QdrantDenseRetriever(timed, "embedding_benchmark", client=client)
        for repetition in range(repetitions):
            started = perf_counter()
            retriever.index(chunks)
            index_ms.append((perf_counter() - started) * 1_000)
            if repetition == 0:
                for _ in range(warmup_runs):
                    for query in queries:
                        retriever.retrieve(query.question, top_k=top_k)
                timed.query_ms.clear()
            runs.append(evaluate_retriever(retriever, queries, max_k=top_k, cutoffs=cutoffs))
    finally:
        client.close()
    documents = summarize_latency(timed.document_ms)
    return EmbeddingBenchmarkResult(
        profile=profile,
        model_metadata=model_metadata or {},
        model_load_ms=round(model_load_ms, 6),
        chunk_count=len(chunks),
        query_count=len(queries),
        dimension=embedder.dimension,
        raw_float32_vector_bytes=len(chunks) * embedder.dimension * 4,
        repetitions=repetitions,
        warmup_runs=warmup_runs,
        document_embedding=documents,
        index_build=summarize_latency(index_ms),
        query_embedding=summarize_latency(timed.query_ms),
        retrieval_latency=summarize_latency(
            [query.latency_ms for run in runs for query in run.per_query]
        ),
        document_embeddings_per_second=round(
            len(chunks) * 1_000 / documents.mean_ms if documents.mean_ms else 0.0, 6
        ),
        retrieval_runs=runs,
    )
