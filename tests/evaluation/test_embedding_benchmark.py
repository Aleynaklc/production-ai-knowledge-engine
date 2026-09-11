"""Controlled benchmark tests use tiny deterministic vectors, never model downloads."""

import math

import pytest
from pydantic import ValidationError

from backend.app.evaluation.dataset import RetrievalEvaluationQuery
from backend.app.evaluation.embedding import (
    EmbeddingProfile,
    benchmark_embedding,
    summarize_latency,
)
from backend.app.evaluation.metrics import evaluate_retriever
from backend.app.ingestion.models import DocumentChunk
from backend.app.retrieval.models import RetrievalResult
from scripts.benchmark_embeddings import build_parser, render_markdown


class CountingEmbedder:
    dimension = 2

    def __init__(self) -> None:
        self.document_calls = 0
        self.query_calls = 0

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.document_calls += 1
        return [[1.0, 0.0] if "alpha" in text else [0.0, 1.0] for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return [1.0, 0.0] if "alpha" in text else [0.0, 1.0]


def _chunks() -> list[DocumentChunk]:
    return [
        DocumentChunk(
            chunk_id=f"00000000-0000-0000-0000-{i:012d}",
            document_id=f"doc-{i}",
            source=f"doc-{i}.md",
            text=text,
            chunk_index=0,
        )
        for i, text in enumerate(["alpha passage", "beta passage"], start=1)
    ]


def _queries(chunks: list[DocumentChunk]) -> list[RetrievalEvaluationQuery]:
    return [
        RetrievalEvaluationQuery(
            id="q1",
            question="alpha",
            category="direct",
            relevant_document_ids=[chunks[0].document_id],
            relevant_chunk_ids=[chunks[0].chunk_id],
        ),
        RetrievalEvaluationQuery(id="q2", question="unknown", category="unanswerable"),
    ]


def _profile() -> EmbeddingProfile:
    return EmbeddingProfile(name="fake", model_name="fake", revision="a" * 40)


def test_embedding_benchmark_excludes_warmups_and_keeps_repeated_evidence() -> None:
    chunks = _chunks()
    embedder = CountingEmbedder()
    report = benchmark_embedding(
        embedder,
        _profile(),
        chunks,
        _queries(chunks),
        repetitions=2,
        warmup_runs=1,
        top_k=1,
        model_load_ms=20,
    )
    assert embedder.document_calls == 3
    assert embedder.query_calls == 6
    assert len(report.document_embedding.samples_ms) == 2
    assert len(report.query_embedding.samples_ms) == 4
    assert len(report.retrieval_latency.samples_ms) == 4
    assert len(report.index_build.samples_ms) == 2
    assert report.raw_float32_vector_bytes == 16
    assert report.model_load_ms == 20
    assert all(run.metrics.recall_at_k == {"1": 1.0} for run in report.retrieval_runs)
    assert report.retrieval_runs[0].metrics.answerable_query_count == 1
    assert "Stage 23" in render_markdown([report], 1, "test")
    assert len(type(report).model_validate_json(report.model_dump_json()).retrieval_runs) == 2


def test_invalid_labels_fail_before_embedding_work() -> None:
    chunks = _chunks()
    queries = _queries(chunks)
    queries[0] = queries[0].model_copy(update={"relevant_chunk_ids": ["missing"]})
    embedder = CountingEmbedder()
    with pytest.raises(ValueError, match="relevance labels"):
        benchmark_embedding(embedder, _profile(), chunks, queries)
    assert embedder.document_calls == embedder.query_calls == 0


@pytest.mark.parametrize("repetitions,warmups,k", [(0, 1, 5), (1, -1, 5), (1, 0, 0)])
def test_invalid_run_configuration(repetitions: int, warmups: int, k: int) -> None:
    chunks = _chunks()
    with pytest.raises(ValueError):
        benchmark_embedding(
            CountingEmbedder(),
            _profile(),
            chunks,
            _queries(chunks),
            repetitions=repetitions,
            warmup_runs=warmups,
            top_k=k,
        )


def test_profiles_require_immutable_revision_and_valid_batch_size() -> None:
    with pytest.raises(ValidationError):
        EmbeddingProfile(name="floating", model_name="model", revision="main")
    with pytest.raises(ValidationError):
        EmbeddingProfile(name="bad-batch", model_name="model", revision="a" * 40, batch_size=0)


def test_percentiles_and_nonfinite_samples() -> None:
    summary = summarize_latency([1, 2, 3, 4, 10])
    assert summary.mean_ms == 4
    assert summary.p50_ms == 3
    assert summary.p95_ms == 8.8
    for values in ([], [-1.0], [math.inf], [math.nan]):
        with pytest.raises(ValueError):
            summarize_latency(values)


class DuplicateRetriever:
    name = "duplicates"

    def __init__(self, chunk: DocumentChunk) -> None:
        self.chunk = chunk

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        return [
            RetrievalResult(
                chunk=self.chunk,
                rank=index,
                score=1.0,
                retriever=self.name,
            )
            for index in range(1, top_k + 1)
        ]


def test_recall_differs_from_hit_rate_and_duplicate_results_get_no_extra_credit() -> None:
    chunks = _chunks()
    queries = [
        RetrievalEvaluationQuery(
            id="multi",
            question="both",
            category="multi_hop",
            relevant_document_ids=[chunk.document_id for chunk in chunks],
            relevant_chunk_ids=[chunk.chunk_id for chunk in chunks],
        )
    ]
    metrics = evaluate_retriever(DuplicateRetriever(chunks[0]), queries).metrics
    assert metrics.recall_at_k["5"] == 0.5
    assert metrics.hit_rate_at_k["5"] == 1.0
    assert metrics.precision_at_k["5"] == 0.2
    assert metrics.ndcg_at_k["5"] == pytest.approx(1 / (1 + 1 / math.log2(3)), abs=1e-6)


@pytest.mark.parametrize("cutoffs", [(), (0,), (-1,), (1, 1), (6,)])
def test_invalid_cutoffs(cutoffs: tuple[int, ...]) -> None:
    chunks = _chunks()
    with pytest.raises(ValueError):
        evaluate_retriever(DuplicateRetriever(chunks[0]), _queries(chunks), cutoffs=cutoffs)


def test_cli_rejects_zero_repetitions() -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--repetitions", "0"])
