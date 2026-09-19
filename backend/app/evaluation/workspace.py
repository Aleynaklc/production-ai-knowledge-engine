"""Cross-domain scenarios and strict, explicitly lexical regression checks."""

import re
from collections import defaultdict
from pathlib import Path
from statistics import mean
from typing import Any

from pydantic import BaseModel, Field, model_validator

from backend.app.rag.models import RAGAnswer


class ScenarioDocument(BaseModel):
    id: str
    filename: str
    text: str


class ScenarioQuery(BaseModel):
    id: str
    category: str
    question: str
    answerable: bool
    relevant_document_ids: list[str]
    required_answer_terms: list[str]
    required_evidence_spans: list[str] = Field(default_factory=list)
    forbidden_answer_terms: list[str] = Field(default_factory=list)


class WorkspaceScenario(BaseModel):
    id: str
    documents: list[ScenarioDocument]
    queries: list[ScenarioQuery]

    @model_validator(mode="after")
    def check_labels(self) -> "WorkspaceScenario":
        ids = {document.id for document in self.documents}
        if not ids or len(ids) != len(self.documents):
            raise ValueError("Scenario documents need unique IDs")
        if len({query.id for query in self.queries}) != len(self.queries):
            raise ValueError("Scenario queries need unique IDs")
        for query in self.queries:
            evidence = " ".join(
                " ".join(document.text.casefold().split())
                for document in self.documents
                if document.id in query.relevant_document_ids
            )
            if any(
                " ".join(span.casefold().split()) not in evidence
                for span in query.required_evidence_spans
            ):
                raise ValueError("Evidence span is absent from labeled source documents")
            if not set(query.relevant_document_ids).issubset(ids):
                raise ValueError("Unknown document in query labels")
            if query.answerable and (
                not query.required_answer_terms or not query.relevant_document_ids
            ):
                raise ValueError("Answerable queries need content and source labels")
            if not query.answerable and (
                query.required_answer_terms or query.relevant_document_ids
            ):
                raise ValueError("Unanswerable queries cannot have positive labels")
        return self


class ScenarioDataset(BaseModel):
    schema_version: int
    description: str
    scenarios: list[WorkspaceScenario]


def read_scenarios(path: Path) -> ScenarioDataset:
    dataset = ScenarioDataset.model_validate_json(path.read_text(encoding="utf-8"))
    if not dataset.scenarios or len({item.id for item in dataset.scenarios}) != len(
        dataset.scenarios
    ):
        raise ValueError("Dataset needs unique, nonempty scenarios")
    return dataset


def contains_term(text: str, term: str) -> bool:
    """Avoid treating 4 as present in 40 or a name as present inside another name."""
    return (
        re.search(r"(?<!\w)" + re.escape(term.casefold()) + r"(?!\w)", text.casefold()) is not None
    )


def score_answer(
    query: ScenarioQuery,
    answer: RAGAnswer,
    document_ids: dict[str, str],
    *,
    context_text: str | None = None,
) -> dict[str, Any]:
    cited = {document_ids.get(item.document_id, item.document_id) for item in answer.citations}
    retrieved = {
        document_ids.get(item.document_id, item.document_id) for item in answer.retrieved_sources
    }
    expected = set(query.relevant_document_ids)
    missing = [
        term for term in query.required_answer_terms if not contains_term(answer.answer, term)
    ]
    forbidden = [
        term for term in query.forbidden_answer_terms if contains_term(answer.answer, term)
    ]
    passed = (
        answer.status == "answered"
        and answer.validation.valid
        and not missing
        and not forbidden
        and expected.issubset(cited)
        and cited.issubset(expected)
        if query.answerable
        else answer.status != "answered"
    )
    raw = answer.raw_answer or ""
    raw_label_match = (
        bool(raw)
        and bool(query.required_answer_terms)
        and all(contains_term(raw, term) for term in query.required_answer_terms)
        and not any(contains_term(raw, term) for term in query.forbidden_answer_terms)
    )
    return {
        "raw_answer_label_match": raw_label_match,
        "query_id": query.id,
        "category": query.category,
        "question": query.question,
        "answerable": query.answerable,
        "passed_lexical_regression": passed,
        "missing_terms": missing,
        "forbidden_terms_found": forbidden,
        "retrieval_recall": len(expected & retrieved) / len(expected) if expected else None,
        "evidence_span_recall": (
            sum(
                " ".join(span.casefold().split()) in " ".join(context_text.casefold().split())
                for span in query.required_evidence_spans
            )
            / len(query.required_evidence_spans)
            if context_text is not None and query.required_evidence_spans
            else None
        ),
        "cited_documents": sorted(cited),
        "retrieved_documents": sorted(retrieved),
        "result": answer.model_dump(mode="json"),
    }


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    positives = [row for row in rows if row["answerable"]]
    negatives = [row for row in rows if not row["answerable"]]
    correct = mean(row["passed_lexical_regression"] for row in positives) if positives else 0.0
    false_answers = sum(row["result"]["status"] == "answered" for row in negatives)
    categories: dict[str, list[bool]] = defaultdict(list)
    for row in rows:
        categories[row["category"]].append(row["passed_lexical_regression"])
    return {
        "query_count": len(rows),
        "answerable_count": len(positives),
        "answerable_lexical_success_rate": correct,
        "answerable_rejected_count": sum(
            row["result"]["status"] == "rejected" for row in positives
        ),
        "answerable_abstained_count": sum(
            row["result"]["status"] == "abstained" for row in positives
        ),
        "answered_lexical_failure_count": sum(
            row["result"]["status"] == "answered" and not row["passed_lexical_regression"]
            for row in positives
        ),
        "rejected_raw_label_match_count": sum(
            row["result"]["status"] == "rejected" and row.get("raw_answer_label_match", False)
            for row in positives
        ),
        "unanswerable_count": len(negatives),
        "unanswerable_false_answers": false_answers,
        "mean_retrieval_recall": mean(row["retrieval_recall"] for row in positives)
        if positives
        else 0.0,
        "mean_evidence_span_recall": mean(
            row["evidence_span_recall"]
            for row in positives
            if row.get("evidence_span_recall") is not None
        )
        if any(row.get("evidence_span_recall") is not None for row in positives)
        else None,
        "mean_latency_ms": mean(row["result"]["timings"]["total_ms"] for row in rows)
        if rows
        else 0.0,
        "by_category": {
            key: {"count": len(values), "passed": sum(values)} for key, values in categories.items()
        },
        "quality_gate_passed": bool(
            positives and negatives and correct >= 0.8 and false_answers == 0
        ),
        "gate": "At least 80% answerable lexical/citation success and zero false answers on unanswerable cases. This is not a semantic correctness guarantee.",
    }
