"""Uploaded evidence reaches both retrieval paths without reloading the language model."""

import re
from pathlib import Path
from typing import cast

import pytest
from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.documents.library import DocumentLibrary
from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.pipeline import write_chunks
from backend.app.llm.generation import GenerationOptions, GenerationResult
from backend.app.rag.context import ContextBuilder
from backend.app.rag.factory import LazyRAGService, RAGRuntime
from backend.app.rag.service import RAGService
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import RerankedRetriever


class WordCodec:
    def count(self, text: str) -> int:
        return len(self.offsets(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return [(match.start(), match.end()) for match in re.finditer(r"\S+", text)]


class KeywordEmbedder:
    dimension = 2
    fail = False

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if self.fail:
            raise RuntimeError("indexing failed")
        return [self.embed_query(text) for text in texts]

    def embed_query(self, text: str) -> list[float]:
        return [1.0, 0.0] if "Borealis" in text else [0.0, 1.0]


class Scorer:
    def score(self, query: str, passages: list[str]) -> list[float]:
        return [10.0 if "Borealis" in passage else 1.0 for passage in passages]


class Generator:
    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        return GenerationResult(
            text="Borealis support code is BLUE-7429 [S1].",
            input_tokens=40,
            output_tokens=10,
            generation_seconds=0.01,
            tokens_per_second=1000,
            options=GenerationOptions(do_sample=False),
        )


def _base() -> DocumentChunk:
    return DocumentChunk(
        chunk_id="00000000-0000-0000-0000-000000000001",
        document_id="base",
        source="base.md",
        text="General system documentation.",
        chunk_index=0,
    )


def _runtime(settings: Settings, chunks: list[DocumentChunk]) -> RAGRuntime:
    client = QdrantClient(location=":memory:")
    embedder = KeywordEmbedder()
    dense = QdrantDenseRetriever(embedder, "initial", client=client)
    dense.index(chunks)
    generator = Generator()
    service = RAGService(dense, ContextBuilder(WordCodec(), token_budget=150), generator)
    return RAGRuntime(service, client, embedder, Scorer(), settings, "initial")


def test_upload_refreshes_dense_and_bm25_and_preserves_generator(tmp_path: Path) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3")
    library = DocumentLibrary(settings, codec=WordCodec())
    record = library.upload("borealis.txt", b"Borealis support code is BLUE-7429.")
    runtime = _runtime(settings, [_base()])
    generator = runtime.service.generator
    try:
        runtime.refresh([_base(), *library.snapshot()[1]])
        ranked = cast(RerankedRetriever, runtime.service.retriever)
        hybrid = cast(HybridRetriever, ranked.base)
        assert hybrid.dense.retrieve("Borealis", 1)[0].chunk.document_id == record.document.id
        sparse_ids = {item.chunk.document_id for item in hybrid.sparse.retrieve("Borealis", 5)}
        assert record.document.id in sparse_ids
        assert runtime.service.generator is generator
        answer = runtime.service.answer("What is the Borealis support code?")
        assert answer.status == "answered"
        assert answer.citations[0].document_id == record.document.id
        assert not runtime.qdrant_client.collection_exists("initial")
    finally:
        runtime.close()


def test_failed_refresh_keeps_previous_index_and_service(tmp_path: Path) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3")
    runtime = _runtime(settings, [_base()])
    previous = runtime.service.retriever
    embedder = cast(KeywordEmbedder, runtime.embedder)
    embedder.fail = True
    try:
        with pytest.raises(RuntimeError, match="indexing failed"):
            runtime.refresh([_base()])
        assert runtime.collection_name == "initial"
        assert runtime.service.retriever is previous
        assert runtime.qdrant_client.collection_exists("initial")
        assert previous.retrieve("General", 1)[0].chunk.document_id == "base"
    finally:
        runtime.close()


def test_uploaded_only_corpus_without_english_tokens_keeps_dense_retrieval(tmp_path: Path) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3")
    library = DocumentLibrary(settings, codec=WordCodec())
    record = library.upload("guide.txt", "日本語の案内文。".encode())
    runtime = _runtime(settings, [_base()])
    try:
        runtime.refresh(library.snapshot()[1])
        ranked = cast(RerankedRetriever, runtime.service.retriever)
        hybrid = cast(HybridRetriever, ranked.base)
        assert hybrid.sparse.retrieve("案内") == []
        assert ranked.retrieve("案内", 1)[0].chunk.document_id == record.document.id
    finally:
        runtime.close()


def test_lazy_service_observes_uploads_without_rebuilding_models(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    settings = Settings(documents_path=tmp_path / "library.sqlite3")
    write_chunks(tmp_path / "data/processed/chunks_recursive.jsonl", [_base()])
    library = DocumentLibrary(settings, codec=WordCodec())
    built: list[RAGRuntime] = []

    def build(settings: Settings, project_root: Path, *, chunks: list[DocumentChunk]) -> RAGRuntime:
        runtime = _runtime(settings, chunks)
        built.append(runtime)
        return runtime

    monkeypatch.setattr("backend.app.rag.factory.build_rag_runtime", build)
    service = LazyRAGService(settings, document_library=library, project_root=tmp_path)
    try:
        service.answer("General question")
        record = library.upload("borealis.txt", b"Borealis support code is BLUE-7429.")
        answer = service.answer("What is the Borealis support code?")
        assert len(built) == 1
        assert answer.status == "answered"
        assert answer.citations[0].document_id == record.document.id
        active_collection = built[0].collection_name
        library.upload("copy.txt", b"Borealis support code is BLUE-7429.")
        service.answer("What is the Borealis support code?")
        assert built[0].collection_name == active_collection
        assert len(built) == 1
    finally:
        service.close()
