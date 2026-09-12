"""CLI failures must not load models or overwrite previously valid corpus files."""

import re
from pathlib import Path

import pytest

from backend.app.evaluation.dataset import RetrievalEvaluationQuery, SourceEvaluationQuery
from backend.app.ingestion.chunkers import HuggingFaceTokenCodec
from backend.app.ingestion.models import DocumentChunk
from scripts import ingest, rag, search


class WordCodec:
    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


@pytest.mark.parametrize("command", [rag, search])
@pytest.mark.parametrize("arguments", [["question", "--top-k", "0"], ["   "]])
def test_invalid_query_fails_before_runtime_loading(
    monkeypatch: pytest.MonkeyPatch, command: object, arguments: list[str]
) -> None:
    monkeypatch.setattr("sys.argv", ["command", *arguments])
    with pytest.raises(SystemExit) as error:
        if command is rag:
            rag.main()
        else:
            search.main()
    assert error.value.code == 2


def test_later_chunking_validation_failure_keeps_all_previous_outputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    raw = tmp_path / "raw"
    raw.mkdir()
    (raw / "guide.md").write_text("# Guide\n\nSupport information.")
    processed = tmp_path / "data/processed"
    processed.mkdir(parents=True)
    original = processed / "chunks_fixed.jsonl"
    original.write_text("previous valid corpus\n")
    queries = tmp_path / "data/evaluation/retrieval_queries_source.jsonl"
    queries.parent.mkdir()
    queries.write_text("")
    calls = 0

    def resolve(
        source_queries: list[SourceEvaluationQuery], chunks: list[DocumentChunk]
    ) -> list[RetrievalEvaluationQuery]:
        nonlocal calls
        calls += 1
        if calls == 2:
            raise ValueError("second strategy cannot resolve labels")
        return []

    monkeypatch.setattr(ingest, "PROJECT_ROOT", tmp_path)
    monkeypatch.setattr(ingest, "resolve_queries", resolve)
    monkeypatch.setattr(HuggingFaceTokenCodec, "from_pretrained", lambda *args: WordCodec())
    monkeypatch.setattr("sys.argv", ["ingest", str(raw)])
    with pytest.raises(ValueError, match="second strategy"):
        ingest.main()
    assert original.read_text() == "previous valid corpus\n"
    assert not (queries.parent / "retrieval_queries_fixed.jsonl").exists()


def test_ingestion_rejects_empty_source_before_loading_tokenizer(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("sys.argv", ["ingest", str(tmp_path)])
    with pytest.raises(SystemExit) as error:
        ingest.main()
    assert error.value.code == 2
