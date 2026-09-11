"""Atomic storage and resource bounds for user-supplied text."""

import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest

from backend.app.config import Settings
from backend.app.documents.library import DocumentLibrary, DocumentUploadError
from backend.app.rag.context import ContextBuilder
from backend.app.retrieval.models import RetrievalResult


class WordCodec:
    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


def test_reading_empty_library_does_not_create_database(tmp_path: Path) -> None:
    path = tmp_path / "missing" / "library.sqlite3"
    library = DocumentLibrary(Settings(documents_path=path))
    assert library.list_documents() == []
    assert library.snapshot()[1] == []
    assert not path.parent.exists()


def test_bom_normalization_and_source_lineage_survive_restart(tmp_path: Path) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3")
    library = DocumentLibrary(settings, codec=WordCodec())
    uploaded = library.upload(
        "policy.MD", b"\xef\xbb\xbf# Policy\r\n\r\nSupport is available.  \r\n"
    )
    restored = DocumentLibrary(settings)
    assert restored.list_documents() == [uploaded.document]
    revision, chunks = restored.snapshot()
    assert revision == library.snapshot()[0]
    assert chunks[0].source == "uploads/policy.md"
    assert chunks[0].metadata["title"] == "Policy"
    assert "\r" not in chunks[0].text
    assert chunks[0].document_id == uploaded.document.id


def test_parallel_duplicate_uploads_create_exactly_one_record(tmp_path: Path) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3")

    def upload(index: int) -> bool:
        library = DocumentLibrary(settings, codec=WordCodec())
        return library.upload(f"copy-{index}.md", b"# Shared policy\n\nContact support.").duplicate

    with ThreadPoolExecutor(max_workers=4) as executor:
        outcomes = list(executor.map(upload, range(8)))
    assert outcomes.count(False) == 1
    assert outcomes.count(True) == 7
    library = DocumentLibrary(settings)
    assert len(library.list_documents()) == 1
    assert len(library.snapshot()[1]) == library.list_documents()[0].chunk_count


def test_total_chunk_capacity_is_atomic_across_documents(tmp_path: Path) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3", upload_max_chunks=1)
    library = DocumentLibrary(settings, codec=WordCodec())
    first = library.upload("first.txt", b"First policy.")
    snapshot = library.snapshot()
    with pytest.raises(DocumentUploadError) as error:
        library.upload("second.txt", b"Second policy.")
    assert error.value.code == "chunk_limit_exceeded"
    assert library.list_documents() == [first.document]
    assert library.snapshot() == snapshot


def test_processing_failure_does_not_leave_a_partial_upload(tmp_path: Path) -> None:
    class BrokenCodec(WordCodec):
        def count(self, text: str) -> int:
            raise RuntimeError("tokenizer unavailable")

    library = DocumentLibrary(
        Settings(documents_path=tmp_path / "library.sqlite3"), codec=BrokenCodec()
    )
    with pytest.raises(DocumentUploadError) as error:
        library.upload("guide.md", b"# Guide\n\nA document.")
    assert error.value.status_code == 503
    assert library.list_documents() == []
    assert not library.path.exists()


def test_long_heading_does_not_consume_the_entire_answer_context(tmp_path: Path) -> None:
    library = DocumentLibrary(
        Settings(documents_path=tmp_path / "library.sqlite3"), codec=WordCodec()
    )
    heading = "Türkçe heading " * 800
    record = library.upload("long.md", f"# {heading}\n\nSupport policy details.".encode())
    assert len(record.document.title.encode("utf-8")) <= 160
    assert record.document.title.endswith("…")
    chunks = library.snapshot()[1]
    assert all(chunk.metadata["title"] == record.document.title for chunk in chunks)
    assert any("Support policy details." in chunk.text for chunk in chunks)
    context = ContextBuilder(WordCodec(), token_budget=650).build(
        "What is the policy?",
        [RetrievalResult(chunk=chunks[-1], score=10, rank=1, retriever="test")],
    )
    assert context.sources
    assert "Support policy details." in context.sources[0].text


def test_heading_with_large_internal_whitespace_is_parsed_without_backtracking(
    tmp_path: Path,
) -> None:
    library = DocumentLibrary(
        Settings(documents_path=tmp_path / "library.sqlite3"), codec=WordCodec()
    )
    record = library.upload("spaces.md", ("# a" + " " * 4_000 + "x ##\n\nPolicy.").encode())
    assert record.document.title == "a x"
