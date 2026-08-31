"""Composition root for the fully local grounded RAG runtime."""

from dataclasses import dataclass
from pathlib import Path
from threading import Lock

from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from backend.app.ingestion.chunkers import HuggingFaceTokenCodec
from backend.app.ingestion.pipeline import read_chunks
from backend.app.llm.generation import GenerationOptions
from backend.app.llm.model import load_runtime
from backend.app.rag.context import ContextBuilder
from backend.app.rag.fallback import ExtractiveFallback
from backend.app.rag.generator import LocalGroundedGenerator
from backend.app.rag.models import RAGAnswer
from backend.app.rag.service import RAGService
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.reranker import CrossEncoderScorer, RerankedRetriever
from backend.app.retrieval.sparse import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[3]


@dataclass(slots=True)
class RAGRuntime:
    """RAG service and the storage handle that owns its local Qdrant lock."""

    service: RAGService
    qdrant_client: QdrantClient

    def close(self) -> None:
        self.qdrant_client.close()


def build_rag_runtime(settings: Settings, project_root: Path = PROJECT_ROOT) -> RAGRuntime:
    """Load pinned local models and compose the production retrieval/generation path."""

    chunks = read_chunks(project_root / "data/processed/chunks_recursive.jsonl")
    embedding = SentenceTransformerEmbedder(
        settings.embedding_model_name,
        settings.embedding_model_revision,
        settings.retrieval_device,
    )
    qdrant_path = settings.qdrant_path
    if not qdrant_path.is_absolute():
        qdrant_path = project_root / qdrant_path
    qdrant_client = QdrantClient(path=str(qdrant_path))
    dense = QdrantDenseRetriever(
        embedding,
        f"{settings.qdrant_collection}_recursive",
        client=qdrant_client,
    )
    if not qdrant_client.collection_exists(dense.collection_name):
        dense.index(chunks)
    bm25 = BM25Retriever(chunks)
    hybrid = HybridRetriever(dense, bm25, settings.retrieval_candidate_k)
    reranker = CrossEncoderScorer(
        settings.reranker_model_name,
        settings.reranker_model_revision,
        settings.retrieval_device,
    )
    retriever = RerankedRetriever(
        hybrid,
        reranker,
        settings.retrieval_candidate_k,
    )

    language_model = load_runtime(
        settings.model_name,
        settings.model_revision,
        settings.device,
    )
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
    return RAGRuntime(service=service, qdrant_client=qdrant_client)


class LazyRAGService:
    """Delay expensive model loading until the first API request."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self._runtime: RAGRuntime | None = None
        self._lock = Lock()

    def _get_runtime(self) -> RAGRuntime:
        if self._runtime is None:
            with self._lock:
                if self._runtime is None:
                    self._runtime = build_rag_runtime(self.settings)
        return self._runtime

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        return self._get_runtime().service.answer(question, top_k)

    def close(self) -> None:
        if self._runtime is not None:
            self._runtime.close()
