"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from pathlib import Path
from typing import Literal

from pydantic import AliasChoices, Field, SecretStr, field_validator, model_validator
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
    generation_provider: Literal["auto", "local", "openai"] = "auto"
    openai_api_key: SecretStr | None = Field(
        default=None,
        repr=False,
        exclude=True,
        validation_alias=AliasChoices("OPENAI_API_KEY", "PAKE_OPENAI_API_KEY"),
    )
    openai_model: str = Field(
        default="gpt-4.1-mini", min_length=1, max_length=160, pattern=r"^[A-Za-z0-9._:/-]+$"
    )
    openai_timeout_seconds: float = Field(default=30.0, ge=1, le=180, allow_inf_nan=False)
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
    retrieval_candidate_k: int = Field(default=60, ge=2, le=200)
    rag_top_k: int = Field(default=5, ge=1, le=20)
    rag_max_sources: int = Field(default=20, ge=1, le=20)
    rag_context_tokens: int = Field(default=1600, ge=128, le=16_384)
    rag_max_new_tokens: int = Field(default=320, ge=16, le=2_048)
    rag_strict_grounding: bool = True
    # Cross-encoder logits rank passages; they are not calibrated answerability scores.
    rag_min_retrieval_score: float | None = None
    # Optional relative reranker-logit window, not a probability/answerability cutoff.
    rag_rerank_score_gap: float | None = Field(default=None, gt=0, allow_inf_nan=False)
    rag_extractive_fallback_score: float = 1.0
    rag_preload_on_startup: bool = True
    rag_cache_max_entries: int = Field(default=256, ge=0, le=10_000)
    rag_cache_ttl_seconds: int = Field(default=900, ge=1, le=86_400)
    api_cors_origins: str = (
        "http://localhost:3000,http://127.0.0.1:3000,http://localhost:3001,http://127.0.0.1:3001,"
        "http://localhost:5173,http://127.0.0.1:5173"
    )
    trace_max_records: int = Field(default=200, ge=1, le=10_000)
    documents_path: Path = Path("data/uploads/documents.sqlite3")
    upload_max_bytes: int = Field(default=5 * 1024 * 1024, ge=1, le=50 * 1024 * 1024)
    upload_max_documents: int = Field(default=100, ge=1, le=10_000)
    upload_max_chunks: int = Field(default=2_000, ge=1, le=100_000)
    workspaces_path: Path = Path("data/workspaces")
    auth_session_seconds: int = Field(default=28_800, ge=60, le=604_800)
    auth_allow_registration: bool = True
    upload_max_expanded_bytes: int = Field(default=20 * 1024 * 1024, ge=1)
    upload_max_pages: int = Field(default=500, ge=1, le=5_000)
    upload_max_versions: int = Field(default=10, ge=1, le=100)
    upload_ocr_enabled: bool = True
    upload_ocr_languages: str = Field(default="eng", pattern=r"^[A-Za-z_]+(?:\+[A-Za-z_]+)*$")

    @field_validator("openai_api_key", mode="before")
    @classmethod
    def normalize_api_key(cls, value: object) -> object:
        if isinstance(value, SecretStr):
            value = value.get_secret_value()
        if isinstance(value, str):
            return value.strip() or None
        return value

    @property
    def effective_generation_provider(self) -> Literal["local", "openai"]:
        if self.generation_provider == "auto":
            return "openai" if self.openai_api_key else "local"
        return self.generation_provider

    @property
    def generation_model(self) -> str:
        return (
            self.openai_model if self.effective_generation_provider == "openai" else self.model_name
        )

    @model_validator(mode="after")
    def validate_chunk_window(self) -> "Settings":
        if self.generation_provider == "openai" and not self.openai_api_key:
            raise ValueError("OpenAI generation requires OPENAI_API_KEY on the backend")
        if self.chunk_overlap_tokens >= self.chunk_size_tokens:
            raise ValueError("chunk_overlap_tokens must be smaller than chunk_size_tokens")
        return self

    @property
    def cors_origins(self) -> list[str]:
        """Return normalized browser origins accepted by the API."""

        return [origin.strip() for origin in self.api_cors_origins.split(",") if origin.strip()]

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAKE_",
        extra="ignore",
        populate_by_name=True,
        hide_input_in_errors=True,
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
