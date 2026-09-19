"""Public contracts for grounded RAG answers."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from backend.app.rag.calculation import CalculationEvidence
from backend.app.rag.citations import CitationValidation
from backend.app.rag.extraction import EvidenceExtraction

type RAGStatus = Literal["answered", "abstained", "rejected"]


class AnswerCitation(BaseModel):
    """A cited source with enough lineage for UI rendering and auditing."""

    model_config = ConfigDict(frozen=True)

    citation_id: str
    chunk_id: str
    document_id: str
    source: str
    title: str
    snippet: str
    supporting_chunk_ids: list[str] = Field(default_factory=list)
    retrieval_score: float
    document_version: int | None = None
    source_unit: int | None = None
    source_kind: str | None = None


class RAGTiming(BaseModel):
    """Wall-clock timings for each visible RAG phase."""

    model_config = ConfigDict(frozen=True)

    retrieval_ms: float = Field(ge=0)
    context_ms: float = Field(ge=0)
    generation_ms: float = Field(ge=0)
    grounding_ms: float = Field(ge=0)
    total_ms: float = Field(ge=0)


class RAGAnswer(BaseModel):
    """Safe user answer plus raw output and grounding evidence."""

    model_config = ConfigDict(frozen=True)

    question: str
    cache_hit: bool = False
    status: RAGStatus
    answer: str
    raw_answer: str | None
    fallback_used: bool
    calculation: CalculationEvidence | None = None
    extraction: EvidenceExtraction | None = None
    citations: list[AnswerCitation]
    retrieved_sources: list[AnswerCitation] = Field(default_factory=list)
    outcome_reason: (
        Literal["no_evidence", "not_in_sources", "verification_failed", "generation_limit"] | None
    ) = None
    validation: CitationValidation
    context_source_count: int = Field(ge=0)
    context_token_count: int = Field(ge=0)
    timings: RAGTiming
    input_tokens: int = Field(ge=0)
    output_tokens: int = Field(ge=0)
    generation_provider: Literal["local", "openai"] | None = None
    generation_model: str | None = None
