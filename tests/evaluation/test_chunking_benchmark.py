"""Chunking benchmark coverage with synthetic vectors and actual isolated Qdrant indexes."""

import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import pytest
from qdrant_client import QdrantClient

import backend.app.evaluation.chunking as chunking_module
from backend.app.evaluation.chunking import (
    ChunkingBenchmarkConfig,
    ChunkingBenchmarkResult,
    ChunkingVariant,
    benchmark_chunking,
    chunking_markdown,
)
from backend.app.evaluation.dataset import SourceEvaluationQuery
from backend.app.ingestion.models import RawDocument


class WordCodec:
    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


class KeywordEmbedder:
    """Synthetic test fixture; never used to produce benchmark documentation."""

    def __init__(self) -> None:
        self.indexed_batches: list[list[str]] = []
        self.query_calls = 0

    @property
    def dimension(self) -> int:
        return 3

    def _vector(self, text: str) -> list[float]:
        return [float("alpha" in text), float("beta" in text), 0.1]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        self.indexed_batches.append(texts)
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        self.query_calls += 1
        return self._vector(text)


def _documents() -> list[RawDocument]:
    return [
        RawDocument(
            document_id="guide",
            source="guide.md",
            title="Guide",
            text="alpha red.\n\nOther filler.\n\nbeta blue.",
        ),
    ]


def _queries() -> list[SourceEvaluationQuery]:
    return [
        SourceEvaluationQuery(
            id="q1",
            question="alpha",
            category="direct",
            relevant_document_ids=["guide"],
            relevant_texts=["alpha red"],
        ),
        SourceEvaluationQuery(
            id="q2",
            question="beta",
            category="semantic",
            relevant_document_ids=["guide"],
            relevant_texts=["beta blue"],
        ),
        SourceEvaluationQuery(id="q3", question="unknown", category="unanswerable"),
    ]


def _variants() -> list[ChunkingVariant]:
    return [
        ChunkingVariant(strategy="fixed", chunk_size_tokens=2, overlap_tokens=0),
        ChunkingVariant(strategy="recursive", chunk_size_tokens=6, overlap_tokens=1),
    ]


def test_variants_rechunk_and_resolve_labels_without_reusing_index() -> None:
    embedder = KeywordEmbedder()
    result = benchmark_chunking(
        _documents(),
        _queries(),
        WordCodec(),
        embedder,
        _variants(),
        config=ChunkingBenchmarkConfig(top_k=2, repetitions=2, warmup_runs=1),
        embedding_token_count=WordCodec().count,
        embedding_max_tokens=4,
    )

    assert [item.chunk_count for item in result.variants] == [3, 1]
    assert len(embedder.indexed_batches) == 2
    assert embedder.query_calls == 2 * (1 + 3 * 2)
    fixed, recursive = result.variants
    assert fixed.resolved_queries[0].relevant_chunk_ids != (
        recursive.resolved_queries[0].relevant_chunk_ids
    )
    assert fixed.embedding_truncated_chunk_count == 0
    assert recursive.embedding_truncated_chunk_count == 1
    assert fixed.raw_float32_vector_bytes == 3 * embedder.dimension * 4
    assert fixed.chunk_tokens_total == result.source_token_count == 6
    assert fixed.token_expansion_ratio == 1.0
    assert len(fixed.trials) == 2
    for variant in result.variants:
        assert variant.trials[0].metrics.answerable_query_count == 2
        assert variant.trials[0].metrics.unanswerable_query_count == 1
        assert variant.trials[0].metrics.recall_at_k["2"] == 1.0
        assert variant.query_latency_p95_ms >= variant.query_latency_p50_ms >= 0
        assert variant.index_build_ms >= variant.embedding_documents_ms >= 0
        assert variant.ingestion_ms >= variant.chunking_ms >= 0
    assert "Recall@2" in chunking_markdown(result)
    assert "Hit rate@2" in chunking_markdown(result)
    assert ChunkingBenchmarkResult.model_validate_json(result.model_dump_json()) == result


def test_fingerprints_reproduce_and_change_with_input() -> None:
    def run(documents: list[RawDocument]) -> ChunkingBenchmarkResult:
        return benchmark_chunking(
            documents,
            _queries(),
            WordCodec(),
            KeywordEmbedder(),
            _variants(),
            config=ChunkingBenchmarkConfig(repetitions=1, warmup_runs=0),
        )

    first = run(_documents())
    second = run(_documents())
    changed = run([_documents()[0].model_copy(update={"text": _documents()[0].text + " Extra."})])
    assert first.source_sha256 == second.source_sha256
    assert first.labels_sha256 == second.labels_sha256 == changed.labels_sha256
    assert first.source_sha256 != changed.source_sha256
    assert first.variants[0].resolved_queries == second.variants[0].resolved_queries


def test_latency_percentiles_pool_all_measured_queries_and_exclude_warmups(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ticks = iter(
        timestamp for index in range(6) for timestamp in (float(index), index + (index + 1) / 1_000)
    )
    monkeypatch.setattr("backend.app.evaluation.metrics.perf_counter", lambda: next(ticks))
    embedder = KeywordEmbedder()

    report = benchmark_chunking(
        _documents(),
        _queries(),
        WordCodec(),
        embedder,
        [_variants()[0]],
        config=ChunkingBenchmarkConfig(repetitions=2, warmup_runs=2),
    )

    variant = report.variants[0]
    assert embedder.query_calls == 8
    assert [row.latency_ms for trial in variant.trials for row in trial.per_query] == [
        1.0,
        2.0,
        3.0,
        4.0,
        5.0,
        6.0,
    ]
    assert variant.query_latency_mean_ms == 3.5
    assert variant.query_latency_p50_ms == 3.5
    assert variant.query_latency_p95_ms == 5.75


def test_all_indexes_are_memory_only_and_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    clients: list[QdrantClient] = []

    def memory_client(*args: Any, **kwargs: Any) -> QdrantClient:
        assert args == ()
        assert kwargs == {"location": ":memory:"}
        client = QdrantClient(**kwargs)
        clients.append(client)
        return client

    monkeypatch.setattr(chunking_module, "QdrantClient", memory_client)
    benchmark_chunking(
        _documents(),
        _queries(),
        WordCodec(),
        KeywordEmbedder(),
        _variants(),
        config=ChunkingBenchmarkConfig(repetitions=1, warmup_runs=0),
    )
    assert len(clients) == 2
    for client in clients:
        with pytest.raises(RuntimeError, match="closed"):
            client.get_collections()


def test_index_is_closed_when_embedding_fails(monkeypatch: pytest.MonkeyPatch) -> None:
    client = QdrantClient(location=":memory:")
    monkeypatch.setattr(chunking_module, "QdrantClient", lambda **_: client)

    class FailingEmbedder(KeywordEmbedder):
        def embed_documents(self, texts: list[str]) -> list[list[float]]:
            raise RuntimeError("fixture embedding error")

    with pytest.raises(RuntimeError, match="fixture embedding error"):
        benchmark_chunking(_documents(), _queries(), WordCodec(), FailingEmbedder(), _variants())
    with pytest.raises(RuntimeError, match="closed"):
        client.get_collections()


@pytest.mark.parametrize("size,overlap", [(0, 0), (-1, 0), (4, -1), (4, 4), (4, 5)])
def test_invalid_chunk_windows_fail(size: int, overlap: int) -> None:
    with pytest.raises(ValueError):
        ChunkingVariant(strategy="fixed", chunk_size_tokens=size, overlap_tokens=overlap)


@pytest.mark.parametrize("field,value", [("top_k", 0), ("repetitions", 0), ("warmup_runs", -1)])
def test_invalid_timing_controls_fail(field: str, value: int) -> None:
    with pytest.raises(ValueError):
        ChunkingBenchmarkConfig.model_validate({field: value})


@pytest.mark.parametrize(
    "counter,maximum",
    [(WordCodec().count, None), (None, 4), (WordCodec().count, 0), (WordCodec().count, -1)],
)
def test_invalid_embedding_token_diagnostics_fail_before_indexing(
    counter: Callable[[str], int] | None,
    maximum: int | None,
) -> None:
    embedder = KeywordEmbedder()
    with pytest.raises(ValueError, match="Embedding"):
        benchmark_chunking(
            _documents(),
            _queries(),
            WordCodec(),
            embedder,
            _variants(),
            embedding_token_count=counter,
            embedding_max_tokens=maximum,
        )
    assert embedder.indexed_batches == []


def test_nonpositive_embedding_dimension_fails_before_indexing() -> None:
    class InvalidEmbedder(KeywordEmbedder):
        @property
        def dimension(self) -> int:
            return 0

    with pytest.raises(ValueError, match="dimension must be positive"):
        benchmark_chunking(_documents(), _queries(), WordCodec(), InvalidEmbedder(), _variants())


def test_missing_or_duplicate_inputs_and_unresolved_labels_fail() -> None:
    def run(
        documents: list[RawDocument],
        queries: list[SourceEvaluationQuery],
        variants: list[ChunkingVariant],
    ) -> None:
        benchmark_chunking(documents, queries, WordCodec(), KeywordEmbedder(), variants)

    with pytest.raises(ValueError, match="non-empty documents"):
        run([], _queries(), _variants())
    with pytest.raises(ValueError, match="answerable query"):
        run(_documents(), [_queries()[-1]], _variants())
    with pytest.raises(ValueError, match="unique"):
        run(_documents(), [_queries()[0], _queries()[0]], _variants())
    with pytest.raises(ValueError, match="unique"):
        run(_documents() * 2, _queries(), _variants())
    with pytest.raises(ValueError, match="duplicate variants"):
        run(_documents(), _queries(), [_variants()[0], _variants()[0]])
    bad_query = _queries()[0].model_copy(update={"relevant_texts": ["never present anywhere"]})
    with pytest.raises(ValueError, match="Variant fixed_2_0: Could not resolve"):
        run(_documents(), [bad_query], _variants())


def test_cli_rejects_invalid_window_before_loading_models(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    from scripts.benchmark_chunking import main

    monkeypatch.setattr(
        "sys.argv",
        ["benchmark_chunking", "--sizes", "0", "--output", str(tmp_path / "report.json")],
    )
    with pytest.raises(SystemExit) as exc_info:
        main()
    assert exc_info.value.code == 2
    assert not (tmp_path / "report.json").exists()
