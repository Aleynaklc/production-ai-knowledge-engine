"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the application.

    Environment variables use the ``PAKE_`` prefix. For example,
    ``PAKE_LOG_LEVEL=DEBUG`` overrides ``log_level``.
    """

    app_name: str = "Production AI Knowledge Engine"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    model_name: str = "Qwen/Qwen2.5-0.5B-Instruct"
    model_revision: str = "c89bee90d9f811437d9735454613c35b4a3c4dc8"
    device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    max_new_tokens: int = Field(default=128, ge=1, le=2_048)
    chunk_strategy: Literal["fixed", "recursive"] = "recursive"
    chunk_size_tokens: int = Field(default=160, ge=32, le=1_024)
    chunk_overlap_tokens: int = Field(default=30, ge=0, le=256)
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    embedding_model_revision: str = "1110a243fdf4706b3f48f1d95db1a4f5529b4d41"
    reranker_model_name: str = "cross-encoder/ms-marco-MiniLM-L6-v2"
    reranker_model_revision: str = "233902d25c440f23af6f7d6e94d2946bac0bee0a"
    retrieval_device: Literal["auto", "cpu", "mps", "cuda"] = "auto"
    qdrant_path: Path = Path("data/qdrant")
    qdrant_collection: str = "novastack_chunks"
    retrieval_top_k: int = Field(default=5, ge=1, le=100)
    retrieval_candidate_k: int = Field(default=20, ge=2, le=200)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_max_sources: int = Field(default=3, ge=1, le=20)
    rag_context_tokens: int = Field(default=650, ge=128, le=16_384)
    rag_max_new_tokens: int = Field(default=80, ge=16, le=2_048)
    rag_strict_grounding: bool = True
    rag_min_retrieval_score: float = 0.8
    rag_extractive_fallback_score: float = 1.0
    api_cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:5173,http://127.0.0.1:5173"
    )
    trace_max_records: int = Field(default=200, ge=1, le=10_000)
    documents_path: Path = Path("data/uploads/documents.sqlite3")
    upload_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    upload_max_documents: int = Field(default=100, ge=1, le=10_000)
    upload_max_chunks: int = Field(default=2_000, ge=1, le=100_000)

    @property
    def cors_origins(self) -> list[str]:
        """Return normalized browser origins accepted by the API."""

        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAKE_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
