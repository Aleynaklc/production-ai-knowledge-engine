"""Deterministic ingestion and chunk-label tests."""

import re
from pathlib import Path

from backend.app.evaluation.dataset import SourceEvaluationQuery, resolve_queries
from backend.app.ingestion.chunkers import FixedTokenChunker, RecursiveChunker
from backend.app.ingestion.cleaner import clean_text
from backend.app.ingestion.models import RawDocument
from backend.app.ingestion.parser import load_documents
from backend.app.ingestion.pipeline import build_chunks


class WordCodec:
    """Small test codec that treats whitespace-separated words as tokens."""

    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def test_clean_text_normalizes_newlines_and_blank_space() -> None:
    assert clean_text("A  \r\n\r\n\r\n B\t\r\n") == "A\n\n B"


def test_loader_is_sorted_and_extracts_markdown_title(tmp_path: Path) -> None:
    (tmp_path / "z.txt").write_text("last", encoding="utf-8")
    (tmp_path / "a.md").write_text("# First\n\nbody", encoding="utf-8")

    documents = load_documents(tmp_path)

    assert [document.document_id for document in documents] == ["a", "z"]
    assert documents[0].title == "First"
    assert documents[0].source == "a.md"


def test_fixed_chunker_enforces_size_and_overlap() -> None:
    chunker = FixedTokenChunker(WordCodec(), chunk_size=4, overlap=1)

    chunks = chunker.chunk("one two three four five six seven")

    assert chunks == ["one two three four", "four five six seven"]
    assert all(chunker.codec.count(chunk) <= 4 for chunk in chunks)


def test_recursive_chunker_preserves_blocks_when_they_fit() -> None:
    chunker = RecursiveChunker(WordCodec(), chunk_size=7, overlap=1)

    chunks = chunker.chunk("# Heading\n\nAlpha beta.\n\nGamma delta epsilon.")

    assert chunks[0] == "# Heading\n\nAlpha beta.\n\nGamma delta epsilon."
    assert all(chunker.codec.count(chunk) <= 7 for chunk in chunks)


def test_chunk_ids_are_deterministic_and_labels_resolve() -> None:
    document = RawDocument(
        document_id="guide",
        source="guide.md",
        title="Guide",
        text="Alpha beta gamma. Delta epsilon zeta.",
    )
    chunker = FixedTokenChunker(WordCodec(), chunk_size=4, overlap=1)

    first = build_chunks([document], chunker)
    second = build_chunks([document], chunker)
    query = SourceEvaluationQuery(
        id="q1",
        question="Where is epsilon?",
        category="direct",
        relevant_document_ids=["guide"],
        relevant_texts=["Delta epsilon"],
    )
    resolved = resolve_queries([query], first)

    assert [chunk.chunk_id for chunk in first] == [chunk.chunk_id for chunk in second]
    assert resolved[0].relevant_chunk_ids == [first[1].chunk_id]
