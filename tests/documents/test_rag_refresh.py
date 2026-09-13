"""Uploaded evidence reaches both retrieval paths without reloading the language model."""

import re
from collections.abc import Iterator
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import cast

import pytest
from fastapi.testclient import TestClient
from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.documents.library import DocumentLibrary
from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.pipeline import write_chunks
from backend.app.llm.generation import GenerationOptions, GenerationResult
from backend.app.main import create_app
from backend.app.rag.cache import AnswerCache
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
    calls = 0

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        self.calls += 1
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


@pytest.fixture
def cached_service(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[tuple[LazyRAGService, DocumentLibrary, list[RAGRuntime]]]:
    settings = Settings(
        documents_path=tmp_path / "library.sqlite3",
        rag_cache_max_entries=2,
        rag_cache_ttl_seconds=10,
        rag_preload_on_startup=True,
    )
    write_chunks(tmp_path / "data/processed/chunks_recursive.jsonl", [_base()])
    library = DocumentLibrary(settings, codec=WordCodec())
    library.upload("borealis.txt", b"Borealis support code is BLUE-7429.")
    built: list[RAGRuntime] = []

    def build(settings: Settings, project_root: Path, *, chunks: list[DocumentChunk]) -> RAGRuntime:
        runtime = _runtime(settings, chunks)
        built.append(runtime)
        return runtime

    monkeypatch.setattr("backend.app.rag.factory.build_rag_runtime", build)
    service = LazyRAGService(settings, document_library=library, project_root=tmp_path)
    try:
        yield service, library, built
    finally:
        service.close()


type CachedService = tuple[LazyRAGService, DocumentLibrary, list[RAGRuntime]]
QUESTION = "What is the Borealis support code?"


def test_cache_reuses_validated_answer_and_isolates_mutable_fields(
    cached_service: CachedService,
) -> None:
    service, _, built = cached_service
    first = service.answer(QUESTION)
    assert first.status == "answered" and not first.cache_hit
    first.citations.clear()
    cached = service.answer(f"  {QUESTION}  ")
    assert cached.cache_hit and cached.citations
    cached.citations.clear()
    assert service.answer(QUESTION).citations
    assert cached.timings.generation_ms == cached.output_tokens == cached.input_tokens == 0
    assert cast(Generator, built[0].service.generator).calls == 1
    assert not service.answer(QUESTION, top_k=1).cache_hit
    assert not service.answer(QUESTION.lower()).cache_hit


def test_cache_expires_without_sliding_on_hits(
    cached_service: CachedService, monkeypatch: pytest.MonkeyPatch
) -> None:
    service, _, _ = cached_service
    clock = [100.0]
    monkeypatch.setattr("backend.app.rag.cache.monotonic", lambda: clock[0])
    service.answer(QUESTION)
    clock[0] = 109.0
    assert service.answer(QUESTION).cache_hit
    clock[0] = 110.0
    assert not service.answer(QUESTION).cache_hit


def test_cache_evicts_least_recently_used_answer(cached_service: CachedService) -> None:
    service, _, _ = cached_service
    service.answer(QUESTION, 1)
    service.answer(QUESTION, 2)
    assert service.answer(QUESTION, 1).cache_hit
    service.answer(QUESTION, 3)
    assert service.answer(QUESTION, 1).cache_hit
    assert not service.answer(QUESTION, 2).cache_hit


def test_upload_invalidates_cache_and_duplicate_preserves_it(cached_service: CachedService) -> None:
    service, library, built = cached_service
    service.answer(QUESTION)
    library.upload("copy.txt", b"Borealis support code is BLUE-7429.")
    assert service.answer(QUESTION).cache_hit
    library.upload("new.txt", b"New evidence in the shared library.")
    assert not service.answer(QUESTION).cache_hit
    assert service.answer(QUESTION).cache_hit
    assert len(built) == 1


def test_failed_refresh_never_serves_stale_cache(cached_service: CachedService) -> None:
    service, library, built = cached_service
    service.answer(QUESTION)
    library.upload("new.txt", b"New evidence.")
    embedder = cast(KeywordEmbedder, built[0].embedder)
    embedder.fail = True
    with pytest.raises(RuntimeError, match="indexing failed"):
        service.answer(QUESTION)
    embedder.fail = False
    assert not service.answer(QUESTION).cache_hit


def test_cache_disabled_and_unvalidated_results_are_not_stored(
    cached_service: CachedService,
) -> None:
    service, _, _ = cached_service
    answer = service.answer(QUESTION)
    key = ("revision", QUESTION, 5)
    disabled = AnswerCache(0, 10)
    disabled.put(key, answer)
    assert disabled.get(key) is None
    cache = AnswerCache(2, 10)
    for status in ("rejected", "abstained"):
        cache.put(key, answer.model_copy(update={"status": status}))
        assert cache.get(key) is None
    invalid = answer.model_copy(
        update={
            "validation": answer.validation.model_copy(update={"valid": False}),
        }
    )
    cache.put(key, invalid)
    assert cache.get(key) is None


def test_simultaneous_repeated_questions_generate_once(cached_service: CachedService) -> None:
    service, _, built = cached_service
    with ThreadPoolExecutor(max_workers=4) as pool:
        answers = list(pool.map(service.answer, [QUESTION] * 4))
    assert sum(answer.cache_hit for answer in answers) == 3
    assert cast(Generator, built[0].service.generator).calls == 1


def test_startup_prepares_models_and_cached_requests_get_fresh_traces(
    cached_service: CachedService,
) -> None:
    service, library, built = cached_service
    assert built == []
    app = create_app(service.settings, rag_service=service, document_library=library)
    with TestClient(app) as client:
        assert len(built) == 1
        generator = cast(Generator, built[0].service.generator)
        assert generator.calls == 0
        first = client.post("/api/v1/answers", json={"question": QUESTION}).json()
        second = client.post("/api/v1/answers", json={"question": QUESTION}).json()
        assert generator.calls == 1
        assert second["result"]["cache_hit"] and second["trace"]["cache_hit"]
        assert first["trace_id"] != second["trace_id"]
        assert first["request_id"] != second["request_id"]
        assert all(stage["status"] == "skipped" for stage in second["trace"]["stages"])
        assert second["trace"]["tokens"] == {"context": 0, "input": 0, "output": 0}
        assert second["result"]["citations"] == first["result"]["citations"]
        assert client.get("/api/v1/traces").json()["count"] == 2
    assert service._runtime is None
    service.prepare()
    assert len(built) == 2
    assert not service.answer(QUESTION).cache_hit


def test_preload_can_be_disabled(cached_service: CachedService) -> None:
    service, library, built = cached_service
    settings = service.settings.model_copy(update={"rag_preload_on_startup": False})
    with TestClient(create_app(settings, rag_service=service, document_library=library)) as client:
        assert built == []
        assert client.post("/api/v1/answers", json={"question": QUESTION}).status_code == 200
        assert len(built) == 1


def test_startup_failure_prevents_serving_and_closes_runtime(
    cached_service: CachedService,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    service, library, _ = cached_service
    prepare = LazyRAGService.prepare

    def fail_after_loading(instance: LazyRAGService) -> None:
        prepare(instance)
        raise RuntimeError("startup failed")

    monkeypatch.setattr(LazyRAGService, "prepare", fail_after_loading)
    with pytest.raises(RuntimeError, match="startup failed"):
        with TestClient(
            create_app(service.settings, rag_service=service, document_library=library)
        ):
            pytest.fail("Startup failure must prevent accepting requests")
    assert service._runtime is None
