"""Grounded-answer dataset contracts and auditable Stage 25 RAG metrics."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, JsonValue, model_validator

from backend.app.rag.citations import CitationValidation
from backend.app.rag.models import AnswerCitation, RAGTiming
from backend.app.rag.service import AnswerService


class GroundedEvaluationQuery(BaseModel):
    """Question labels for lexical answer content, abstention, and source correctness."""

    model_config = ConfigDict(frozen=True)

    id: str = Field(min_length=1)
    question: str = Field(min_length=1)
    answerable: bool
    relevant_document_ids: list[str] = Field(default_factory=list)
    required_answer_terms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_labels(self) -> "GroundedEvaluationQuery":
        if not self.id.strip() or not self.question.strip():
            raise ValueError("RAG query IDs and questions cannot be blank")
        if any(
            not item.strip() for item in self.relevant_document_ids + self.required_answer_terms
        ):
            raise ValueError("RAG labels cannot be blank")
        if self.answerable and (not self.relevant_document_ids or not self.required_answer_terms):
            raise ValueError("Answerable RAG queries require source and answer labels")
        if not self.answerable and (self.relevant_document_ids or self.required_answer_terms):
            raise ValueError("Unanswerable RAG queries cannot have positive labels")
        return self


class GroundedQueryEvaluation(BaseModel):
    """Auditable outcome for one measured request, including its returned evidence."""

    model_config = ConfigDict(frozen=True)

    query_id: str
    answerable: bool
    status: str
    answer: str
    raw_answer: str | None
    fallback_used: bool
    answer_terms_present: bool
    cited_document_ids: list[str]
    citation_precision: float
    citation_recall: float
    grounding_valid: bool
    latency_ms: float
    validation_issues: list[str]
    question: str = ""
    repetition: int = Field(default=1, ge=1)
    relevant_document_ids: list[str] = Field(default_factory=list)
    required_answer_terms: list[str] = Field(default_factory=list)
    missing_answer_terms: list[str] = Field(default_factory=list)
    lexical_answer_correct: bool = False
    citations: list[AnswerCitation] = Field(default_factory=list)
    context_source_count: int = 0
    context_token_count: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    timings: RAGTiming | None = None
    validation: CitationValidation | None = None


class RAGLatencySummary(BaseModel):
    """Service-reported phase latency over measured requests, including zero phases."""

    model_config = ConfigDict(frozen=True)

    sample_count: int
    mean_ms: float
    p50_ms: float
    p95_ms: float
    min_ms: float
    max_ms: float


class GroundedRAGMetrics(BaseModel):
    """Aggregate lexical proxy, source, abstention, grounding, and latency metrics."""

    model_config = ConfigDict(frozen=True)

    query_count: int
    answerable_query_count: int
    unanswerable_query_count: int
    answer_correctness: float
    citation_precision: float
    citation_recall: float
    grounded_answer_rate: float
    answerable_response_rate: float
    abstention_accuracy: float
    safe_unanswerable_rate: float
    rejection_rate: float
    extractive_fallback_rate: float
    mean_latency_ms: float
    p95_latency_ms: float
    lexical_answer_correctness: float = 0.0
    unanswerable_false_answer_rate: float = 0.0
    answerable_abstention_rate: float = 0.0
    answerable_rejection_rate: float = 0.0
    unanswerable_rejection_rate: float = 0.0
    grounded_lexical_answer_rate: float = 0.0
    status_counts: dict[str, int] = Field(default_factory=dict)


class GroundedRAGEvaluation(BaseModel):
    """Stage 25 report; legacy Stage 13–16 metric names remain readable."""

    model_config = ConfigDict(frozen=True)

    metrics: GroundedRAGMetrics
    per_query: list[GroundedQueryEvaluation]
    schema_version: int = 2
    dataset_query_count: int = 0
    repetitions: int = 1
    warmup_runs: int = 0
    warmup_generation_count: int = 0
    latency_by_stage: dict[str, RAGLatencySummary] = Field(default_factory=dict)
    metadata: dict[str, JsonValue] = Field(default_factory=dict)


def read_rag_queries(path: Path) -> list[GroundedEvaluationQuery]:
    """Read the versioned JSONL grounded-answer dataset."""

    queries = [
        GroundedEvaluationQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    validate_rag_queries(queries)
    return queries


def validate_rag_queries(queries: list[GroundedEvaluationQuery]) -> None:
    """Fail before expensive inference if labels cannot support the aggregate metrics."""

    if not queries:
        raise ValueError("RAG evaluation requires at least one query")
    if len({query.id for query in queries}) != len(queries):
        raise ValueError("RAG evaluation query IDs must be unique")
    if not any(query.answerable for query in queries) or all(query.answerable for query in queries):
        raise ValueError("RAG evaluation requires answerable and unanswerable queries")


def _rate(values: list[bool]) -> float:
    return round(sum(values) / len(values), 6) if values else 0.0


def _latency_summary(values: list[float]) -> RAGLatencySummary:
    return RAGLatencySummary(
        sample_count=len(values),
        mean_ms=round(float(np.mean(values)), 6),
        p50_ms=round(float(np.percentile(values, 50)), 6),
        p95_ms=round(float(np.percentile(values, 95)), 6),
        min_ms=round(min(values), 6),
        max_ms=round(max(values), 6),
    )


def evaluate_rag(
    service: AnswerService,
    queries: list[GroundedEvaluationQuery],
    *,
    warmup_runs: int = 0,
    repetitions: int = 1,
    metadata: dict[str, JsonValue] | None = None,
) -> GroundedRAGEvaluation:
    """Score each full-dataset repeat after excluded warmups, in fixed dataset order.

    Correctness is a case-insensitive substring proxy on *returned, answered* text;
    it is neither a semantic answer judge nor a proof of factual correctness.
    Warmups cycle through answerable questions to exercise the generation path.
    ``query_count`` counts measured calls; ``dataset_query_count`` counts unique labels.
    """

    validate_rag_queries(queries)
    if repetitions < 1 or warmup_runs < 0:
        raise ValueError("repetitions must be positive and warmup_runs must be nonnegative")
    warmup_queries = [query for query in queries if query.answerable]
    warmup_generation_count = 0
    for index in range(warmup_runs):
        warmup = service.answer(warmup_queries[index % len(warmup_queries)].question)
        warmup_generation_count += int(warmup.output_tokens > 0)

    outcomes: list[GroundedQueryEvaluation] = []
    for repetition in range(1, repetitions + 1):
        for query in queries:
            result = service.answer(query.question)
            normalized_answer = result.answer.casefold()
            missing_terms = [
                term
                for term in query.required_answer_terms
                if term.casefold() not in normalized_answer
            ]
            terms_present = query.answerable and not missing_terms
            cited_documents = list(
                dict.fromkeys(citation.document_id for citation in result.citations)
            )
            relevant = set(query.relevant_document_ids)
            cited = set(cited_documents)
            precision = len(cited & relevant) / len(cited) if cited else 0.0
            recall = len(cited & relevant) / len(relevant) if relevant else 0.0
            outcomes.append(
                GroundedQueryEvaluation(
                    query_id=query.id,
                    question=query.question,
                    repetition=repetition,
                    answerable=query.answerable,
                    status=result.status,
                    answer=result.answer,
                    raw_answer=result.raw_answer,
                    fallback_used=result.fallback_used,
                    answer_terms_present=terms_present,
                    lexical_answer_correct=result.status == "answered" and terms_present,
                    relevant_document_ids=query.relevant_document_ids,
                    required_answer_terms=query.required_answer_terms,
                    missing_answer_terms=missing_terms,
                    cited_document_ids=cited_documents,
                    citation_precision=round(precision, 6),
                    citation_recall=round(recall, 6),
                    grounding_valid=result.validation.valid,
                    latency_ms=result.timings.total_ms,
                    validation_issues=result.validation.issues,
                    citations=result.citations,
                    context_source_count=result.context_source_count,
                    context_token_count=result.context_token_count,
                    input_tokens=result.input_tokens,
                    output_tokens=result.output_tokens,
                    timings=result.timings,
                    validation=result.validation,
                )
            )

    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]
    latencies = _latency_summary([outcome.latency_ms for outcome in outcomes])
    cited_answerable = [outcome for outcome in answerable if outcome.cited_document_ids]
    lexical_correctness = _rate([outcome.lexical_answer_correct for outcome in answerable])
    return GroundedRAGEvaluation(
        dataset_query_count=len(queries),
        warmup_runs=warmup_runs,
        warmup_generation_count=warmup_generation_count,
        repetitions=repetitions,
        metadata=metadata or {},
        latency_by_stage={
            stage.removesuffix("_ms"): _latency_summary(
                [
                    float(getattr(outcome.timings, stage))
                    for outcome in outcomes
                    if outcome.timings is not None
                ]
            )
            for stage in ("retrieval_ms", "context_ms", "generation_ms", "grounding_ms", "total_ms")
        },
        metrics=GroundedRAGMetrics(
            query_count=len(outcomes),
            answerable_query_count=len(answerable),
            unanswerable_query_count=len(unanswerable),
            answer_correctness=lexical_correctness,
            lexical_answer_correctness=lexical_correctness,
            citation_precision=round(
                sum(outcome.citation_precision for outcome in cited_answerable)
                / len(cited_answerable)
                if cited_answerable
                else 0.0,
                6,
            ),
            citation_recall=round(
                sum(outcome.citation_recall for outcome in answerable) / len(answerable), 6
            ),
            grounded_answer_rate=_rate(
                [outcome.status == "answered" and outcome.grounding_valid for outcome in answerable]
            ),
            grounded_lexical_answer_rate=_rate(
                [
                    outcome.lexical_answer_correct and outcome.grounding_valid
                    for outcome in answerable
                ]
            ),
            answerable_response_rate=_rate(
                [outcome.status == "answered" for outcome in answerable]
            ),
            abstention_accuracy=_rate([outcome.status == "abstained" for outcome in unanswerable]),
            safe_unanswerable_rate=_rate(
                [outcome.status in {"abstained", "rejected"} for outcome in unanswerable]
            ),
            unanswerable_false_answer_rate=_rate(
                [outcome.status == "answered" for outcome in unanswerable]
            ),
            answerable_abstention_rate=_rate(
                [outcome.status == "abstained" for outcome in answerable]
            ),
            answerable_rejection_rate=_rate(
                [outcome.status == "rejected" for outcome in answerable]
            ),
            unanswerable_rejection_rate=_rate(
                [outcome.status == "rejected" for outcome in unanswerable]
            ),
            rejection_rate=_rate([outcome.status == "rejected" for outcome in outcomes]),
            extractive_fallback_rate=_rate([outcome.fallback_used for outcome in outcomes]),
            mean_latency_ms=latencies.mean_ms,
            p95_latency_ms=latencies.p95_ms,
            status_counts={
                status: sum(outcome.status == status for outcome in outcomes)
                for status in ("answered", "abstained", "rejected")
            },
        ),
        per_query=outcomes,
    )


def write_rag_evaluation(path: Path, evaluation: GroundedRAGEvaluation) -> None:
    """Persist an auditable machine-readable RAG report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evaluation.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
