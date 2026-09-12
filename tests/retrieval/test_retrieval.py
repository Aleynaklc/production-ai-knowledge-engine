"""Unit and local integration coverage for retrieval stages."""

from pathlib import Path

import pytest
from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.evaluation.dataset import RetrievalEvaluationQuery
from backend.app.evaluation.metrics import evaluate_retriever
from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.pipeline import write_chunks
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.models import RetrievalResult
from backend.app.retrieval.reranker import RerankedRetriever
from backend.app.retrieval.sparse import BM25Retriever
from scripts import search


def make_chunk(index: int, text: str) -> DocumentChunk:
    return DocumentChunk(
        chunk_id=f"00000000-0000-0000-0000-{index:012d}",
        document_id=f"doc-{index}",
        source=f"doc-{index}.md",
        text=text,
        chunk_index=0,
    )


class KeywordEmbedder:
    @property
    def dimension(self) -> int:
        return 2

    def _vector(self, text: str) -> list[float]:
        return [1.0, 0.0] if "alpha" in text.casefold() else [0.0, 1.0]

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text)


class StaticRetriever:
    name = "static"

    def __init__(self, results: list[RetrievalResult]) -> None:
        self.results = results

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        del query
        return self.results[:top_k]


class LengthScorer:
    def score(self, query: str, passages: list[str]) -> list[float]:
        del query
        return [float(len(passage)) for passage in passages]


def ranked(chunk: DocumentChunk, rank: int, retriever: str) -> RetrievalResult:
    return RetrievalResult(
        chunk=chunk,
        score=1.0 / rank,
        rank=rank,
        retriever=retriever,
        component_scores={retriever: 1.0 / rank},
    )


def test_bm25_prioritizes_exact_rare_term() -> None:
    chunks = [
        make_chunk(1, "common words"),
        make_chunk(2, "rare quasar identifier"),
        make_chunk(3, "different ordinary terms"),
    ]

    results = BM25Retriever(chunks).retrieve("quasar", top_k=1)

    assert results[0].chunk.chunk_id == chunks[1].chunk_id


def test_rrf_rewards_chunks_returned_by_both_retrievers() -> None:
    one = make_chunk(1, "one")
    two = make_chunk(2, "two")
    three = make_chunk(3, "three")

    results = reciprocal_rank_fusion(
        [
            [ranked(one, 1, "dense"), ranked(two, 2, "dense")],
            [ranked(two, 1, "bm25"), ranked(three, 2, "bm25")],
        ],
        top_k=3,
    )

    assert results[0].chunk.chunk_id == two.chunk_id
    assert results[0].retriever == "hybrid_rrf"


def test_reranker_reorders_candidate_passages() -> None:
    short = make_chunk(1, "short")
    long = make_chunk(2, "a much longer relevant passage")
    base = StaticRetriever([ranked(short, 1, "base"), ranked(long, 2, "base")])

    results = RerankedRetriever(base, LengthScorer(), candidate_k=2).retrieve("query", top_k=2)

    assert results[0].chunk.chunk_id == long.chunk_id
    assert "reranker" in results[0].component_scores


def test_empty_candidates_do_not_reach_cross_encoder() -> None:
    class UnusedScorer:
        def score(self, query: str, passages: list[str]) -> list[float]:
            raise AssertionError("An empty candidate list must not reach the model")

    assert RerankedRetriever(StaticRetriever([]), UnusedScorer()).retrieve("query") == []


def test_rrf_does_not_reward_duplicates_from_one_retriever() -> None:
    one = make_chunk(1, "one")
    results = reciprocal_rank_fusion(
        [[ranked(one, 1, "dense"), ranked(one, 2, "dense")], [ranked(one, 1, "bm25")]],
        top_k=1,
    )
    assert results[0].score == pytest.approx(2 / 61)


def test_metrics_calculate_hit_precision_and_reciprocal_rank() -> None:
    first = make_chunk(1, "first")
    relevant = make_chunk(2, "relevant")
    retriever = StaticRetriever([ranked(first, 1, "static"), ranked(relevant, 2, "static")])
    query = RetrievalEvaluationQuery(
        id="q1",
        question="question",
        category="direct",
        relevant_document_ids=[relevant.document_id],
        relevant_chunk_ids=[relevant.chunk_id],
    )

    evaluation = evaluate_retriever(retriever, [query], max_k=5)

    assert evaluation.metrics.recall_at_k == {"1": 0.0, "3": 1.0, "5": 1.0}
    assert evaluation.metrics.precision_at_k["5"] == 0.2
    assert evaluation.metrics.mrr_at_k["5"] == 0.5


def test_qdrant_local_dense_index_round_trip(tmp_path: Path) -> None:
    chunks = [make_chunk(1, "alpha document"), make_chunk(2, "beta document")]
    client = QdrantClient(path=str(tmp_path / "qdrant"))
    retriever = QdrantDenseRetriever(KeywordEmbedder(), "test_chunks", client=client)
    try:
        retriever.index(chunks)
        results = retriever.retrieve("alpha", top_k=1)
    finally:
        client.close()

    assert results[0].chunk.chunk_id == chunks[0].chunk_id
    assert results[0].chunk.source == "doc-1.md"


def test_search_cli_reindexes_updated_chunks(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    settings = Settings(qdrant_path=tmp_path / "qdrant")
    monkeypatch.setattr(search, "Settings", lambda: settings)
    monkeypatch.setattr(search, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(search, "SentenceTransformerEmbedder", lambda *args: KeywordEmbedder())
    monkeypatch.setattr("sys.argv", ["search", "alpha", "--method", "dense"])
    path = tmp_path / "data/processed/chunks_recursive.jsonl"
    write_chunks(path, [make_chunk(1, "alpha original policy")])
    search.main()
    capsys.readouterr()
    write_chunks(path, [make_chunk(1, "alpha updated policy")])
    search.main()
    result = capsys.readouterr().out
    assert "updated policy" in result
    assert "original policy" not in result
