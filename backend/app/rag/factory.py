"""Composition root for the fully local grounded RAG runtime."""

import hashlib
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from time import perf_counter

from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.documents.library import DocumentLibrary
from backend.app.embeddings.base import Embedder
from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from backend.app.ingestion.chunkers import HuggingFaceTokenCodec, RecursiveChunker
from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.parser import load_documents
from backend.app.ingestion.pipeline import build_chunks, read_chunks
from backend.app.llm.generation import GenerationOptions
from backend.app.llm.model import load_runtime
from backend.app.rag.cache import AnswerCache
from backend.app.rag.context import ContextBuilder
from backend.app.rag.fallback import ExtractiveFallback
from backend.app.rag.generator import LocalGroundedGenerator
from backend.app.rag.models import RAGAnswer, RAGTiming
from backend.app.rag.service import RAGService
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderScorer, PairScorer, RerankedRetriever
from backend.app.retrieval.sparse import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]
logger = logging.getLogger(__name__)


def _base_chunks(settings: Settings, project_root: Path) -> list[DocumentChunk]:
    """Reuse prepared base chunks, or prepare the demo corpus on first use."""

    path = project_root / "data/processed/chunks_recursive.jsonl"
    if path.is_file():
        return read_chunks(path)
    documents = load_documents(project_root / "data/raw")
    if not documents:
        return []
    codec = HuggingFaceTokenCodec.from_pretrained(settings.model_name, settings.model_revision)
    return build_chunks(
        documents,
        RecursiveChunker(codec, settings.chunk_size_tokens, settings.chunk_overlap_tokens),
    )


def _collection_name(settings: Settings, chunks: list[DocumentChunk]) -> str:
    fingerprint = hashlib.sha256()
    fingerprint.update(
        f"{settings.embedding_model_name}@{settings.embedding_model_revision}".encode()
    )
    for chunk in chunks:
        fingerprint.update(chunk.model_dump_json().encode("utf-8"))
        fingerprint.update(b"\n")
    return f"{settings.qdrant_collection}_serving_{fingerprint.hexdigest()[:24]}"


def _build_retriever(
    settings: Settings,
    client: QdrantClient,
    embedding: Embedder,
    scorer: PairScorer,
    chunks: list[DocumentChunk],
    collection: str,
) -> RerankedRetriever:
    """Build a complete replacement before switching the live service to it."""

    sparse = BM25Retriever(chunks)
    dense = QdrantDenseRetriever(embedding, collection, client=client)
    try:
        dense.index(chunks)
    except Exception:
        if client.collection_exists(collection):
            client.delete_collection(collection)
        raise
    hybrid = HybridRetriever(dense, sparse, settings.retrieval_candidate_k)
    return RerankedRetriever(hybrid, scorer, settings.retrieval_candidate_k)


@dataclass(slots=True)
class RAGRuntime:
    """RAG service and the storage handle that owns its local Qdrant lock."""

    service: RAGService
    qdrant_client: QdrantClient
    embedder: Embedder
    scorer: PairScorer
    settings: Settings
    collection_name: str

    def refresh(self, chunks: list[DocumentChunk]) -> None:
        """Refresh both dense and lexical retrieval while reusing all loaded models."""

        collection = _collection_name(self.settings, chunks)
        if collection == self.collection_name:
            return
        replacement = _build_retriever(
            self.settings, self.qdrant_client, self.embedder, self.scorer, chunks, collection
        )
        previous = self.collection_name
        self.service.retriever = replacement
        self.collection_name = collection
        # These are this process's generated serving indexes; evaluation indexes are untouched.
        try:
            self.qdrant_client.delete_collection(previous)
        except Exception:
            logger.warning("A superseded serving index could not be removed", exc_info=True)

    def close(self) -> None:
        self.qdrant_client.close()


def build_rag_runtime(
    settings: Settings,
    project_root: Path = PROJECT_ROOT,
    *,
    chunks: list[DocumentChunk] | None = None,
) -> RAGRuntime:
    """Load pinned local models and compose the production retrieval/generation path."""

    if chunks is None:
        _, uploaded_chunks = DocumentLibrary(settings, project_root).snapshot()
        chunks = _base_chunks(settings, project_root) + uploaded_chunks
    if not chunks:
        raise ValueError("The knowledge base has no documents yet")
    embedding = SentenceTransformerEmbedder(
        settings.embedding_model_name,
        settings.embedding_model_revision,
        settings.retrieval_device,
    )
    qdrant_path = settings.qdrant_path
    if not qdrant_path.is_absolute():
        qdrant_path = project_root / qdrant_path
    qdrant_client = QdrantClient(path=str(qdrant_path))
    collection = _collection_name(settings, chunks)
    try:
        reranker = CrossEncoderScorer(
            settings.reranker_model_name,
            settings.reranker_model_revision,
            settings.retrieval_device,
        )
        retriever = _build_retriever(
            settings, qdrant_client, embedding, reranker, chunks, collection
        )
        language_model = load_runtime(settings.model_name, settings.model_revision, settings.device)
    except Exception:
        qdrant_client.close()
        raise
    codec = HuggingFaceTokenCodec(language_model.tokenizer)
    context_builder = ContextBuilder(
        codec,
        token_budget=settings.rag_context_tokens,
        max_sources=settings.rag_max_sources,
    )
    generator = LocalGroundedGenerator(
        language_model,
        GenerationOptions(
            do_sample=False,
            max_new_tokens=settings.rag_max_new_tokens,
            repetition_penalty=1.05,
        ),
    )
    service = RAGService(
        retriever,
        context_builder,
        generator,
        default_top_k=settings.rag_top_k,
        strict_grounding=settings.rag_strict_grounding,
        minimum_retrieval_score=settings.rag_min_retrieval_score,
        extractive_fallback=ExtractiveFallback(
            reranker,
            minimum_score=settings.rag_extractive_fallback_score,
        ),
    )
    return RAGRuntime(
        service=service,
        qdrant_client=qdrant_client,
        embedder=embedding,
        scorer=reranker,
        settings=settings,
        collection_name=collection,
    )


class LazyRAGService:
    """Own reusable models and a corpus-aware cache; prepare eagerly during API startup."""

    def __init__(
        self,
        settings: Settings,
        *,
        document_library: DocumentLibrary | None = None,
        project_root: Path = PROJECT_ROOT,
    ) -> None:
        # Runtime settings and the cache share one immutable-by-convention snapshot.
        # Configuration changes take effect in a new service/process, never a stale cache.
        self.settings = settings.model_copy(deep=True)
        self.project_root = project_root
        self.document_library = document_library or DocumentLibrary(settings, project_root)
        self._base: list[DocumentChunk] | None = None
        self._document_revision: str | None = None
        self._runtime: RAGRuntime | None = None
        self._lock = Lock()
        self._cache = AnswerCache(settings.rag_cache_max_entries, settings.rag_cache_ttl_seconds)

    def _get_runtime(self) -> RAGRuntime:
        """Caller holds the lock so generation, refresh, and shutdown cannot race."""

        revision, uploaded = self.document_library.snapshot()
        if self._base is None:
            self._base = _base_chunks(self.settings, self.project_root)
        if self._runtime is None:
            self._runtime = build_rag_runtime(
                self.settings, self.project_root, chunks=self._base + uploaded
            )
            self._document_revision = revision
        elif revision != self._document_revision:
            self._cache.clear()
            self._runtime.refresh(self._base + uploaded)
            self._document_revision = revision
        return self._runtime

    def prepare(self) -> None:
        """Load all models and build the current index without generating a user answer."""
        with self._lock:
            self._get_runtime()

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        started = perf_counter()
        normalized = question.strip()
        depth = self.settings.rag_top_k if top_k is None else top_k
        if not normalized or depth < 1:
            raise ValueError("Question must not be blank and top_k must be positive")
        with self._lock:
            # Always check persistent document revision before consulting the cache.
            runtime = self._get_runtime()
            key = (self._document_revision or "", normalized, depth)
            cached = self._cache.get(key)
            if cached is not None:
                return cached.model_copy(
                    update={
                        "cache_hit": True,
                        "input_tokens": 0,
                        "output_tokens": 0,
                        "timings": RAGTiming(
                            retrieval_ms=0,
                            context_ms=0,
                            generation_ms=0,
                            grounding_ms=0,
                            total_ms=(perf_counter() - started) * 1_000,
                        ),
                    }
                )
            answer = runtime.service.answer(normalized, depth)
            answer = answer.model_copy(
                update={
                    "timings": answer.timings.model_copy(
                        update={
                            "total_ms": (perf_counter() - started) * 1_000,
                        }
                    ),
                }
            )
            self._cache.put(key, answer)
            return answer

    def close(self) -> None:
        with self._lock:
            self._cache.clear()
            if self._runtime is not None:
                self._runtime.close()
                self._runtime = None
                self._document_revision = None
