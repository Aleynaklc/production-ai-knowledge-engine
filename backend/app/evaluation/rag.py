"""Grounded-answer dataset contracts and end-to-end RAG metrics."""

import json
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from backend.app.rag.service import AnswerService


class GroundedEvaluationQuery(BaseModel):
    """Question labels for answer content, abstention, and source correctness."""

    model_config = ConfigDict(frozen=True)

    id: str
    question: str
    answerable: bool
    relevant_document_ids: list[str] = Field(default_factory=list)
    required_answer_terms: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def validate_labels(self) -> "GroundedEvaluationQuery":
        if self.answerable and (not self.relevant_document_ids or not self.required_answer_terms):
            raise ValueError("Answerable RAG queries require source and answer labels")
        if not self.answerable and (self.relevant_document_ids or self.required_answer_terms):
            raise ValueError("Unanswerable RAG queries cannot have positive labels")
        return self


class GroundedQueryEvaluation(BaseModel):
    """Auditable outcome for one RAG question."""

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


class GroundedRAGMetrics(BaseModel):
    """Aggregate answer, source, abstention, grounding, and latency metrics."""

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


class GroundedRAGEvaluation(BaseModel):
    """Complete Stage 13–16 benchmark report."""

    model_config = ConfigDict(frozen=True)

    metrics: GroundedRAGMetrics
    per_query: list[GroundedQueryEvaluation]


def read_rag_queries(path: Path) -> list[GroundedEvaluationQuery]:
    """Read the versioned JSONL grounded-answer dataset."""

    return [
        GroundedEvaluationQuery.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def evaluate_rag(
    service: AnswerService,
    queries: list[GroundedEvaluationQuery],
) -> GroundedRAGEvaluation:
    """Run end-to-end RAG and score content, citations, grounding, and abstention."""

    if not queries:
        raise ValueError("RAG evaluation requires at least one query")
    outcomes: list[GroundedQueryEvaluation] = []
    for query in queries:
        result = service.answer(query.question)
        normalized_answer = result.answer.casefold()
        terms_present = query.answerable and all(
            term.casefold() in normalized_answer for term in query.required_answer_terms
        )
        cited_documents = list(dict.fromkeys(citation.document_id for citation in result.citations))
        relevant = set(query.relevant_document_ids)
        cited = set(cited_documents)
        citation_precision = len(cited & relevant) / len(cited) if cited else 0.0
        citation_recall = len(cited & relevant) / len(relevant) if relevant else 0.0
        outcomes.append(
            GroundedQueryEvaluation(
                query_id=query.id,
                answerable=query.answerable,
                status=result.status,
                answer=result.answer,
                raw_answer=result.raw_answer,
                fallback_used=result.fallback_used,
                answer_terms_present=terms_present,
                cited_document_ids=cited_documents,
                citation_precision=round(citation_precision, 6),
                citation_recall=round(citation_recall, 6),
                grounding_valid=result.validation.valid,
                latency_ms=result.timings.total_ms,
                validation_issues=result.validation.issues,
            )
        )

    answerable = [outcome for outcome in outcomes if outcome.answerable]
    unanswerable = [outcome for outcome in outcomes if not outcome.answerable]
    if not answerable or not unanswerable:
        raise ValueError("RAG evaluation requires answerable and unanswerable queries")
    latencies = [outcome.latency_ms for outcome in outcomes]
    cited_answerable = [outcome for outcome in answerable if outcome.cited_document_ids]
    return GroundedRAGEvaluation(
        metrics=GroundedRAGMetrics(
            query_count=len(outcomes),
            answerable_query_count=len(answerable),
            unanswerable_query_count=len(unanswerable),
            answer_correctness=round(
                float(np.mean([outcome.answer_terms_present for outcome in answerable])), 6
            ),
            citation_precision=round(
                float(np.mean([outcome.citation_precision for outcome in cited_answerable]))
                if cited_answerable
                else 0.0,
                6,
            ),
            citation_recall=round(
                float(np.mean([outcome.citation_recall for outcome in answerable])), 6
            ),
            grounded_answer_rate=round(
                float(
                    np.mean(
                        [
                            outcome.status == "answered" and outcome.grounding_valid
                            for outcome in answerable
                        ]
                    )
                ),
                6,
            ),
            answerable_response_rate=round(
                float(np.mean([outcome.status == "answered" for outcome in answerable])), 6
            ),
            abstention_accuracy=round(
                float(np.mean([outcome.status == "abstained" for outcome in unanswerable])), 6
            ),
            safe_unanswerable_rate=round(
                float(
                    np.mean(
                        [outcome.status in {"abstained", "rejected"} for outcome in unanswerable]
                    )
                ),
                6,
            ),
            rejection_rate=round(
                float(np.mean([outcome.status == "rejected" for outcome in outcomes])), 6
            ),
            extractive_fallback_rate=round(
                float(np.mean([outcome.fallback_used for outcome in outcomes])), 6
            ),
            mean_latency_ms=round(float(np.mean(latencies)), 6),
            p95_latency_ms=round(float(np.percentile(latencies, 95)), 6),
        ),
        per_query=outcomes,
    )


def write_rag_evaluation(path: Path, evaluation: GroundedRAGEvaluation) -> None:
    """Persist a deterministic machine-readable RAG report."""

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(evaluation.model_dump(mode="json"), indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
