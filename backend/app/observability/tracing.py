"""Translate grounded answer diagnostics into a stable system trace."""

from datetime import UTC, datetime
from uuid import uuid4

from backend.app.observability.models import (
    SystemTrace,
    TraceSource,
    TraceStage,
    TraceTokenUsage,
)
from backend.app.rag.models import RAGAnswer


def build_system_trace(answer: RAGAnswer, request_id: str) -> SystemTrace:
    """Build an API-safe trace from timings and lineage already produced by RAG."""

    generated = answer.raw_answer is not None
    trace = SystemTrace(
        trace_id=str(uuid4()),
        request_id=request_id,
        created_at=datetime.now(UTC),
        question=answer.question,
        cache_hit=answer.cache_hit,
        status=answer.status,
        total_ms=answer.timings.total_ms,
        stages=[
            TraceStage(
                name="retrieval",
                status="completed",
                duration_ms=answer.timings.retrieval_ms,
                summary=f"Retrieved evidence for {answer.context_source_count} context sources.",
                metrics={"context_sources": answer.context_source_count},
            ),
            TraceStage(
                name="context_assembly",
                status="completed" if answer.context_source_count else "skipped",
                duration_ms=answer.timings.context_ms,
                summary=(
                    f"Assembled {answer.context_token_count} context tokens."
                    if answer.context_source_count
                    else "No eligible context was assembled."
                ),
                metrics={
                    "context_sources": answer.context_source_count,
                    "context_tokens": answer.context_token_count,
                },
            ),
            TraceStage(
                name="generation",
                status="completed" if generated else "skipped",
                duration_ms=answer.timings.generation_ms,
                summary=(
                    f"Generated {answer.output_tokens} output tokens."
                    if generated
                    else "Computed quantity × unit price from a source table; no model generation."
                    if answer.calculation is not None
                    else "Selected verbatim source excerpts; no model generation."
                    if answer.extraction is not None
                    else "Generation was skipped by the evidence gate."
                ),
                metrics={
                    "input_tokens": answer.input_tokens,
                    "output_tokens": answer.output_tokens,
                    **(
                        {"provider": answer.generation_provider}
                        if answer.generation_provider
                        else {}
                    ),
                    **({"model": answer.generation_model} if answer.generation_model else {}),
                },
            ),
            TraceStage(
                name="grounding",
                status="completed"
                if generated or answer.calculation is not None or answer.extraction is not None
                else "skipped",
                duration_ms=answer.timings.grounding_ms,
                summary=(
                    (
                        "Grounding validation passed."
                        if answer.validation.valid
                        else "Grounding validation prevented an unsafe answer."
                    )
                    if generated or answer.calculation is not None or answer.extraction is not None
                    else "Grounding was skipped because generation did not run."
                ),
                metrics={
                    "validation_valid": answer.validation.valid,
                    "citation_count": len(answer.citations),
                    "fallback_used": answer.fallback_used,
                },
            ),
        ],
        sources=[
            TraceSource(
                citation_id=citation.citation_id,
                chunk_id=citation.chunk_id,
                document_id=citation.document_id,
                source=citation.source,
                title=citation.title,
                retrieval_score=citation.retrieval_score,
            )
            for citation in answer.citations
        ],
        tokens=TraceTokenUsage(
            context=answer.context_token_count,
            input=answer.input_tokens,
            output=answer.output_tokens,
        ),
        validation_issues=answer.validation.issues,
        fallback_used=answer.fallback_used,
    )
    if answer.cache_hit:
        return trace.model_copy(
            update={
                "stages": [
                    stage.model_copy(
                        update={
                            "status": "skipped",
                            "duration_ms": 0.0,
                            "summary": "Reused a cached, validated answer; this stage did not run.",
                            "metrics": {"cache_hit": True},
                        }
                    )
                    for stage in trace.stages
                ],
                "tokens": TraceTokenUsage(context=0, input=0, output=0),
            }
        )
    return trace
