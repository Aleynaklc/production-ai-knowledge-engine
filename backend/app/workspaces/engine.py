"""Shared model weights, isolated retrieval state, and background incremental ingestion."""

import fcntl
import hashlib
import json
import logging
from dataclasses import dataclass
from pathlib import Path
from threading import Event, Lock, Thread
from time import perf_counter
from typing import TextIO
from uuid import NAMESPACE_URL, uuid5

from qdrant_client import QdrantClient, models

from backend.app.config import Settings
from backend.app.documents.library import DocumentUploadError
from backend.app.embeddings.base import Embedder
from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from backend.app.ingestion.chunkers import (
    FixedTokenChunker,
    HuggingFaceTokenCodec,
    RecursiveChunker,
    TokenCodec,
)
from backend.app.ingestion.models import DocumentChunk
from backend.app.llm.generation import GenerationOptions, GenerationResult
from backend.app.llm.model import load_runtime
from backend.app.observability.store import TraceStore
from backend.app.rag.cache import AnswerCache
from backend.app.rag.context import ContextBuilder
from backend.app.rag.fallback import ExtractiveFallback
from backend.app.rag.generator import GroundedGenerator, LocalGroundedGenerator
from backend.app.rag.models import RAGAnswer, RAGTiming
from backend.app.rag.service import RAGService
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderScorer, PairScorer, RerankedRetriever
from backend.app.retrieval.sparse import BM25Retriever
from backend.app.workspaces.auth import AccountStore
from backend.app.workspaces.parsing import extract
from backend.app.workspaces.store import WorkspaceStore

logger = logging.getLogger(__name__)


class LockedEmbedder:
    def __init__(self, base: Embedder) -> None:
        self.base = base
        self.dimension = base.dimension
        self.lock = Lock()

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        result = []
        for offset in range(0, len(texts), 16):
            with self.lock:
                result.extend(self.base.embed_documents(texts[offset : offset + 16]))
        return result

    def embed_query(self, text: str) -> list[float]:
        with self.lock:
            return self.base.embed_query(text)


class LockedScorer:
    def __init__(self, base: PairScorer) -> None:
        self.base = base
        self.lock = Lock()

    def score(self, query: str, passages: list[str]) -> list[float]:
        with self.lock:
            return self.base.score(query, passages)


class LockedGenerator:
    def __init__(self, base: GroundedGenerator) -> None:
        self.base = base
        self.lock = Lock()

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        with self.lock:
            return self.base.generate(prompt, system_prompt)


@dataclass
class SharedModels:
    embedder: Embedder
    scorer: PairScorer
    generator: GroundedGenerator
    codec: TokenCodec

    @classmethod
    def load(cls, settings: Settings) -> "SharedModels":
        embedding = SentenceTransformerEmbedder(
            settings.embedding_model_name,
            settings.embedding_model_revision,
            settings.retrieval_device,
        )
        scorer = CrossEncoderScorer(
            settings.reranker_model_name,
            settings.reranker_model_revision,
            settings.retrieval_device,
        )
        runtime = load_runtime(settings.model_name, settings.model_revision, settings.device)
        return cls(
            LockedEmbedder(embedding),
            LockedScorer(scorer),
            LockedGenerator(
                LocalGroundedGenerator(
                    runtime,
                    GenerationOptions(
                        do_sample=False,
                        max_new_tokens=settings.rag_max_new_tokens,
                        repetition_penalty=1.05,
                    ),
                )
            ),
            HuggingFaceTokenCodec(runtime.tokenizer),
        )


class WorkspaceEngine:
    def __init__(self, store: WorkspaceStore, shared: SharedModels, settings: Settings) -> None:
        self.store, self.shared, self.settings = store, shared, settings
        self.lock = Lock()
        self.cache = AnswerCache(settings.rag_cache_max_entries, settings.rag_cache_ttl_seconds)
        self.traces = TraceStore(settings.trace_max_records)
        self.client = QdrantClient(path=str(store.path.parent / "qdrant"))
        self.collection = "workspace_chunks"
        self.dense = QdrantDenseRetriever(shared.embedder, self.collection, client=self.client)
        self.service: RAGService | None = None
        self.revision = ""
        self.healthy = False
        try:
            self._ensure_embeddings()
            self.restore()
        except Exception:
            self.client.close()
            raise

    def _ensure_embeddings(self) -> None:
        key = f"{self.settings.embedding_model_name}@{self.settings.embedding_model_revision}:{self.shared.embedder.dimension}"
        with self.store.connect() as db:
            previous = db.execute("SELECT value FROM metadata WHERE key='embedding'").fetchone()
            if previous and previous[0] == key:
                return
            rows = db.execute(
                "SELECT document_id,version,chunks FROM versions WHERE chunks!='[]'"
            ).fetchall()
        updates = []
        for row in rows:
            chunks = [DocumentChunk.model_validate(item) for item in json.loads(row[2])]
            vectors = self.shared.embedder.embed_documents([chunk.text for chunk in chunks])
            updates.append((json.dumps(vectors), row[0], row[1]))
        with self.store.connect() as db:
            db.executemany(
                "UPDATE versions SET vectors=? WHERE document_id=? AND version=?", updates
            )
            db.execute("INSERT OR REPLACE INTO metadata VALUES ('embedding', ?)", (key,))

    def _upsert(self, chunks: list[DocumentChunk], vectors: list[list[float]]) -> None:
        if chunks:
            self.client.upsert(
                self.collection,
                points=[
                    models.PointStruct(id=chunk.chunk_id, vector=vector, payload=chunk.model_dump())
                    for chunk, vector in zip(chunks, vectors, strict=True)
                ],
                wait=True,
            )

    def _set_service(self, revision: str, chunks: list[DocumentChunk]) -> None:
        self.revision = revision
        self.cache.clear()
        if not chunks:
            self.service = None
            return
        hybrid = HybridRetriever(
            self.dense, BM25Retriever(chunks), self.settings.retrieval_candidate_k
        )
        self.service = RAGService(
            RerankedRetriever(hybrid, self.shared.scorer, self.settings.retrieval_candidate_k),
            ContextBuilder(
                self.shared.codec,
                token_budget=self.settings.rag_context_tokens,
                max_sources=self.settings.rag_max_sources,
            ),
            self.shared.generator,
            default_top_k=self.settings.rag_top_k,
            strict_grounding=self.settings.rag_strict_grounding,
            minimum_retrieval_score=self.settings.rag_min_retrieval_score,
            extractive_fallback=ExtractiveFallback(
                self.shared.scorer, minimum_score=self.settings.rag_extractive_fallback_score
            ),
        )

    def restore(self) -> None:
        self.healthy = False
        revision, chunks, vectors = self.store.snapshot()
        if self.client.collection_exists(self.collection):
            self.client.delete_collection(self.collection)
        self.client.create_collection(
            self.collection,
            vectors_config=models.VectorParams(
                size=self.shared.embedder.dimension, distance=models.Distance.COSINE
            ),
        )
        self._upsert(chunks, vectors)
        self._set_service(revision, chunks)
        self.healthy = True

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        started = perf_counter()
        question = question.strip()
        depth = self.settings.rag_top_k if top_k is None else top_k
        if not question or depth < 1:
            raise ValueError("Invalid question or top_k")
        with self.lock:
            if not self.healthy:
                self.restore()
            if self.service is None:
                raise DocumentUploadError(
                    "knowledge_base_empty",
                    "No indexed documents yet. Wait for a document to become ready.",
                    409,
                )
            key = (self.revision, question, depth)
            cached = self.cache.get(key)
            if cached:
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
                            total_ms=(perf_counter() - started) * 1000,
                        ),
                    }
                )
            answer = self.service.answer(question, depth)
            answer = answer.model_copy(
                update={
                    "timings": answer.timings.model_copy(
                        update={"total_ms": (perf_counter() - started) * 1000}
                    )
                }
            )
            self.cache.put(key, answer)
            return answer

    def process(self, job: tuple[int, str, int, str, bytes]) -> None:
        job_id, document_id, version, filename, data = job
        try:
            units = extract(filename, data, self.settings)
            chunker_class = (
                FixedTokenChunker if self.settings.chunk_strategy == "fixed" else RecursiveChunker
            )
            chunker = chunker_class(
                self.shared.codec,
                self.settings.chunk_size_tokens,
                self.settings.chunk_overlap_tokens,
            )
            chunks: list[DocumentChunk] = []
            for unit in units:
                for text in chunker.chunk(unit.text):
                    index = len(chunks)
                    chunks.append(
                        DocumentChunk(
                            chunk_id=str(
                                uuid5(
                                    NAMESPACE_URL,
                                    f"{document_id}:{version}:{index}:{hashlib.sha256(text.encode()).hexdigest()}",
                                )
                            ),
                            document_id=document_id,
                            source=filename,
                            text=text,
                            chunk_index=index,
                            metadata={
                                "title": Path(filename).stem,
                                "document_version": version,
                                "source_unit": unit.number,
                                "source_kind": unit.kind,
                            },
                        )
                    )
            if not chunks or len(chunks) > self.settings.upload_max_chunks:
                raise DocumentUploadError(
                    "chunk_limit_exceeded",
                    "Document has no indexable chunks or exceeds the workspace limit.",
                    409,
                )
            vectors = self.shared.embedder.embed_documents([chunk.text for chunk in chunks])
            with self.lock:
                self.store.get(document_id)  # A deletion during parsing must win.
                _, previous, _ = self.store.snapshot()
                retained = [chunk for chunk in previous if chunk.document_id != document_id]
                if len(retained) + len(chunks) > self.settings.upload_max_chunks:
                    raise DocumentUploadError(
                        "chunk_limit_exceeded", "Workspace chunk limit reached.", 409
                    )
                try:
                    self.healthy = False
                    self._upsert(chunks, vectors)
                    old_ids = [
                        chunk.chunk_id for chunk in previous if chunk.document_id == document_id
                    ]
                    if old_ids:
                        self.client.delete(
                            self.collection, models.PointIdsList(points=[*old_ids]), wait=True
                        )
                    if not self.store.complete(
                        job_id, document_id, version, units, chunks, vectors
                    ):
                        self.restore()
                        return
                    revision, active, _ = self.store.snapshot()
                    self._set_service(revision, active)
                    self.healthy = True
                except Exception:
                    self.restore()
                    raise
        except Exception as error:
            logger.exception("Workspace document job failed: %s", job_id)
            message = (
                error.message
                if isinstance(error, DocumentUploadError)
                else "Processing failed. Retry this document."
            )
            self.store.fail(job_id, document_id, version, message)

    def delete(self, document_id: str) -> None:
        with self.lock:
            self.store.get(document_id)
            _, chunks, _ = self.store.snapshot()
            removed = [chunk.chunk_id for chunk in chunks if chunk.document_id == document_id]
            self.store.delete(document_id)
            self.cache.clear()
            self.healthy = False
            try:
                if removed:
                    self.client.delete(
                        self.collection, models.PointIdsList(points=[*removed]), wait=True
                    )
                revision, active, _ = self.store.snapshot()
                self._set_service(revision, active)
                self.healthy = True
            except Exception:
                self.restore()

    def close(self) -> None:
        self.cache.clear()
        self.client.close()


class WorkspaceManager:
    def __init__(self, settings: Settings, shared: SharedModels | None = None) -> None:
        self.settings = settings.model_copy(deep=True)
        self.shared = shared
        self.accounts = AccountStore(
            settings.workspaces_path / "accounts.sqlite3", settings.auth_session_seconds
        )
        self.stores = {
            workspace["id"]: WorkspaceStore(
                settings.workspaces_path / workspace["id"] / "library.sqlite3", settings
            )
            for workspace in self.accounts.workspaces()
        }
        self.engines: dict[str, WorkspaceEngine] = {}
        self.lock = Lock()
        self.stop = Event()
        self.wake = Event()
        self.worker: Thread | None = None
        self.process_lock: TextIO | None = None

    def prepare(self) -> None:
        with self.lock:
            if self.shared is None:
                self.shared = SharedModels.load(self.settings)

    def engine(self, workspace_id: str) -> WorkspaceEngine:
        self.prepare()
        self.store(workspace_id)
        with self.lock:
            if workspace_id not in self.engines:
                assert self.shared is not None
                self.engines[workspace_id] = WorkspaceEngine(
                    self.stores[workspace_id], self.shared, self.settings
                )
            return self.engines[workspace_id]

    def store(self, workspace_id: str) -> WorkspaceStore:
        with self.lock:
            if workspace_id not in self.stores:
                if workspace_id not in {item["id"] for item in self.accounts.workspaces()}:
                    raise DocumentUploadError("workspace_not_found", "Workspace not found.", 404)
                self.stores[workspace_id] = WorkspaceStore(
                    self.settings.workspaces_path / workspace_id / "library.sqlite3", self.settings
                )
            return self.stores[workspace_id]

    def start(self) -> None:
        self.process_lock = (self.settings.workspaces_path / ".runtime.lock").open("a")
        try:
            fcntl.flock(self.process_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            self.process_lock.close()
            self.process_lock = None
            raise RuntimeError(
                "Another API worker is using this workspace directory. Run one worker."
            ) from None
        if self.settings.rag_preload_on_startup:
            self.prepare()
            for workspace_id in self.stores:
                self.engine(workspace_id)
        for store in self.stores.values():
            store.recover()
        self.stop.clear()
        self.worker = Thread(target=self._work, name="workspace-ingestion", daemon=True)
        self.worker.start()

    def _work(self) -> None:
        while not self.stop.is_set():
            self.wake.clear()
            with self.lock:
                stores = list(self.stores.items())
            for workspace_id, store in stores:
                if self.stop.is_set():
                    break
                try:
                    job = store.next_job()
                    if job:
                        try:
                            self.engine(workspace_id).process(job)
                        except Exception:
                            logger.exception("Workspace runtime unavailable")
                            store.fail(
                                job[0],
                                job[1],
                                job[2],
                                "Model preparation failed. Retry this document.",
                            )
                        self.wake.set()
                except Exception:
                    logger.exception("Workspace queue unavailable")
            self.wake.wait(0.5)

    def close(self) -> None:
        self.stop.set()
        self.wake.set()
        if self.worker:
            self.worker.join()
        for engine in self.engines.values():
            engine.close()
        self.engines.clear()
        if self.process_lock:
            fcntl.flock(self.process_lock.fileno(), fcntl.LOCK_UN)
            self.process_lock.close()
            self.process_lock = None
