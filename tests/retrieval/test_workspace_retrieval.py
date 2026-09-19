"""General evidence-selection invariants across languages and arbitrary document names."""

import pytest

from backend.app.retrieval.sparse import BM25Retriever, tokenize_for_bm25
from backend.app.retrieval.workspace import WorkspaceRetriever
from tests.retrieval.test_retrieval import StaticRetriever, make_chunk, ranked


class IntroBiasedScorer:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [0.25 if "Introduction" in passage else 0.0 for passage in passages]


@pytest.mark.parametrize("name", ["Atlas", "Boreal", "Cedar", "Deniz"])
def test_document_scope_and_examples_use_content_not_memorized_questions(name: str) -> None:
    intro = make_chunk(1, "# Introduction\nThis document covers useful operations.").model_copy(
        update={"source": f"{name}-handbook.md", "document_id": name}
    )
    example = make_chunk(
        2, "## Joining records\nExample:\n```sql\nSELECT * FROM events;\n```"
    ).model_copy(update={"source": intro.source, "document_id": name, "chunk_index": 1})
    other = make_chunk(3, "## Other company\nExample: private material.").model_copy(
        update={"source": "Other-handbook.md"}
    )
    base = StaticRetriever([ranked(other, 1, "dense"), ranked(intro, 2, "dense")])
    retriever = WorkspaceRetriever(base, [intro, example, other], IntroBiasedScorer())
    for question in (f"Give examples from {name}", f"{name} örnekleri göster"):
        results = retriever.retrieve(question, 2)
        assert results[0].chunk.chunk_id == example.chunk_id
        assert results[0].chunk.document_id == name
        assert all(
            result.chunk.text in {intro.text, example.text, other.text} for result in results
        )
    assert "section_heading" not in example.metadata  # stored originals remain unchanged


def test_unicode_lexical_tokens_are_not_silently_discarded() -> None:
    assert tokenize_for_bm25("İZİN süresi çığ") == tokenize_for_bm25("izin suresi cig")
    assert tokenize_for_bm25("日本語の案内")
    assert tokenize_for_bm25("+++") == []


def test_bm25_does_not_award_unmatched_chunks_votes_in_hybrid_search() -> None:
    chunks = [make_chunk(1, "alpha alpha"), make_chunk(2, "beta beta")]
    retriever = BM25Retriever(chunks)
    assert retriever.retrieve("unknown") == []
    assert retriever.retrieve("+++") == []
    # Both BM25 scores are zero in this two-document corpus; only alpha matches.
    assert [result.chunk.chunk_id for result in retriever.retrieve("alpha")] == [chunks[0].chunk_id]
    # A single-document corpus can have negative IDF: still a genuine match.
    assert BM25Retriever(chunks[:1]).retrieve("alpha")[0].chunk.chunk_id == chunks[0].chunk_id


def test_comparison_preserves_both_named_documents_and_checks_score_contract() -> None:
    one = make_chunk(1, "# Alpha\nThe limit is 30.").model_copy(update={"source": "Alpha.md"})
    two = make_chunk(2, "# Beta\nThe limit is 60.").model_copy(update={"source": "Beta.md"})
    base = StaticRetriever([ranked(one, 1, "dense"), ranked(two, 2, "dense")])
    retriever = WorkspaceRetriever(base, [one, two], IntroBiasedScorer())
    results = retriever.retrieve("Compare Alpha and Beta limits", 2)
    assert {item.chunk.document_id for item in results} == {one.document_id, two.document_id}
    with pytest.raises(ValueError, match="positive"):
        retriever.retrieve("query", 0)

    class BadScorer:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return []

    with pytest.raises(ValueError, match="score count"):
        WorkspaceRetriever(base, [one, two], BadScorer()).retrieve("query")


def test_filename_hint_does_not_exclude_a_relevant_cv() -> None:
    cv = make_chunk(1, "Python and SQL skills.").model_copy(update={"source": "Person-CV.pdf"})
    code = make_chunk(2, "Python syntax.").model_copy(update={"source": "Python-handbook.md"})

    class RelevantScorer:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [10.0 if "skills" in passage else 0.0 for passage in passages]

    base = StaticRetriever([ranked(cv, 1, "dense"), ranked(code, 2, "dense")])
    results = WorkspaceRetriever(base, [cv, code], RelevantScorer()).retrieve(
        "What Python skills does she have?", 1
    )
    assert results[0].chunk.chunk_id == cv.chunk_id


def test_plural_keyword_search_recovers_the_answer_without_dense_recall() -> None:
    evidence = make_chunk(1, "Aster University and Maple University.")
    assert BM25Retriever([evidence]).retrieve("Which universities?")[0].chunk == evidence


def test_neighbors_keep_page_and_version_lineage() -> None:
    chunks = [
        make_chunk(i + 1, f"part {i}").model_copy(
            update={
                "document_id": "one",
                "chunk_index": i,
                "metadata": {"source_unit": 1 if i < 2 else 2, "document_version": 1},
            }
        )
        for i in range(3)
    ]
    base = StaticRetriever([ranked(chunks[0], 1, "dense")])
    result = WorkspaceRetriever(base, chunks, IntroBiasedScorer()).retrieve("part", 1)[0]
    assert "part 0" in result.chunk.text and "part 1" in result.chunk.text
    assert "part 2" not in result.chunk.text
    assert result.chunk.chunk_id == chunks[0].chunk_id
    assert result.chunk.metadata["supporting_chunk_ids"] == ",".join(c.chunk_id for c in chunks[:2])


@pytest.mark.parametrize("offset", [-30.0, 0.0, 30.0])
def test_relative_score_window_ignores_logit_origin_and_does_not_refill(offset: float) -> None:
    chunks = [make_chunk(i, f"Passage {i}") for i in range(1, 4)]

    class Scorer:
        def score(self, query: str, passages: list[str]) -> list[float]:
            return [
                offset + (0 if "Passage 1" in p else -2 if "Passage 2" in p else -12)
                for p in passages
            ]

    base = StaticRetriever([ranked(chunk, i, "dense") for i, chunk in enumerate(chunks, 1)])
    selected = WorkspaceRetriever(base, chunks, Scorer(), score_gap=6).retrieve("question", 3)
    assert [item.chunk.chunk_id for item in selected] == [chunks[0].chunk_id, chunks[1].chunk_id]
    assert len(WorkspaceRetriever(base, chunks, Scorer()).retrieve("question", 3)) == 3
