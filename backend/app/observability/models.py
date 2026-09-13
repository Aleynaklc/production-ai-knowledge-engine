"""Public contracts for one end-to-end knowledge-engine execution trace."""

from datetime import datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.rag.models import RAGStatus

type TraceStageName = Literal["retrieval", "context_assembly", "generation", "grounding"]
type TraceStageStatus = Literal["completed", "skipped"]


class TraceStage(BaseModel):
    """One measurable phase in the grounded-answer pipeline."""

    model_config = ConfigDict(frozen=True)

    name: TraceStageName
    status: TraceStageStatus
    duration_ms: float = Field(ge=0)
    summary: str
    metrics: dict[str, str | int | float | bool] = Field(default_factory=dict)


class TraceSource(BaseModel):
    """Source lineage returned with an answer and exposed for trace inspection."""

    model_config = ConfigDict(frozen=True)

    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    title: str
    retrieval_score: float


class TraceTokenUsage(BaseModel):
    """Token accounting for context and model input/output."""

    model_config = ConfigDict(frozen=True)

    context: int = Field(ge=0)
    input: int = Field(ge=0)
    output: int = Field(ge=0)


class SystemTrace(BaseModel):
    """Immutable, request-correlated trace stored after one answer attempt."""

    model_config = ConfigDict(frozen=True)

    trace_id: str
    request_id: str
    created_at: datetime
    question: str
    cache_hit: bool = False
    status: RAGStatus
    total_ms: float = Field(ge=0)
    stages: list[TraceStage]
    sources: list[TraceSource]
    tokens: TraceTokenUsage
    validation_issues: list[str]
    fallback_used: bool
