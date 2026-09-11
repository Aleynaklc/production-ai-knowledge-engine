"""Stage 25 regression checks for metric denominators, evidence, and warmup handling."""

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from backend.app.evaluation.rag import (
    GroundedEvaluationQuery,
    GroundedRAGEvaluation,
    evaluate_rag,
    read_rag_queries,
    write_rag_evaluation,
)
from backend.app.rag.citations import CitationValidation
from backend.app.rag.models import AnswerCitation, RAGAnswer, RAGStatus, RAGTiming
from scripts.evaluate_rag import _write_markdown, build_parser, main


def _queries() -> list[GroundedEvaluationQuery]:
    return [
        GroundedEvaluationQuery(
            id="q1",
            question="access",
            answerable=True,
            relevant_document_ids=["a"],
            required_answer_terms=["15 minutes"],
        ),
        GroundedEvaluationQuery(
            id="q2",
            question="payment",
            answerable=True,
            relevant_document_ids=["b"],
            required_answer_terms=["24 hours"],
        ),
        GroundedEvaluationQuery(id="q3", question="unknown", answerable=False),
        GroundedEvaluationQuery(id="q4", question="unsupported", answerable=False),
    ]


def _citation(document: str, index: int) -> AnswerCitation:
    return AnswerCitation(
        citation_id=f"S{index}",
        chunk_id=f"chunk-{index}",
        document_id=document,
        source=f"{document}.md",
        title=document,
        snippet="fixture",
        retrieval_score=1,
    )


def _answer(
    question: str,
    status: RAGStatus,
    answer: str,
    total: float,
    *,
    citations: list[AnswerCitation] | None = None,
    fallback: bool = False,
) -> RAGAnswer:
    evidence = citations or []
    abstained = status == "abstained"
    return RAGAnswer(
        question=question,
        status=status,
        answer=answer,
        raw_answer="fixture raw",
        fallback_used=fallback,
        citations=evidence,
        validation=CitationValidation(
            valid=status != "rejected",
            abstained=abstained,
            cited_source_ids=[item.citation_id for item in evidence],
            unknown_source_ids=[],
            uncited_claims=[],
            unsupported_claims=[],
            issues=["fixture rejection"] if status == "rejected" else [],
        ),
        context_source_count=len(evidence),
        context_token_count=12,
        timings=RAGTiming(
            retrieval_ms=10,
            context_ms=0 if abstained else 1,
            generation_ms=0 if abstained else total - 15,
            grounding_ms=0 if abstained else 2,
            total_ms=total,
        ),
        input_tokens=0 if abstained else 20,
        output_tokens=0 if abstained else 4,
    )


class FixtureService:
    def __init__(self, slow_first: bool = False) -> None:
        self.calls: list[str] = []
        self.slow_first = slow_first
        self.responses = {
            "access": _answer(
                "access",
                "answered",
                "15 MINUTES [S1]",
                40,
                citations=[_citation("a", 1), _citation("a", 2), _citation("wrong", 3)],
                fallback=True,
            ),
            # A rejected result containing every expected term must never score correct.
            "payment": _answer("payment", "rejected", "24 hours", 50),
            "unknown": _answer("unknown", "abstained", "Insufficient evidence.", 10),
            "unsupported": _answer("unsupported", "answered", "Invented answer.", 80),
        }

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        self.calls.append(question)
        response = self.responses[question]
        if self.slow_first and len(self.calls) == 1:
            return response.model_copy(
                update={"timings": response.timings.model_copy(update={"total_ms": 100_000})}
            )
        return response


def test_status_gates_correctness_and_document_citations_are_deduplicated() -> None:
    report = evaluate_rag(FixtureService(), _queries())
    metrics = report.metrics
    assert report.per_query[1].answer_terms_present is True
    assert report.per_query[1].lexical_answer_correct is False
    assert metrics.answer_correctness == metrics.lexical_answer_correctness == 0.5
    assert metrics.grounded_lexical_answer_rate == 0.5
    assert metrics.citation_precision == 0.5
    assert metrics.citation_recall == 0.5
    assert report.per_query[0].cited_document_ids == ["a", "wrong"]
    assert len(report.per_query[0].citations) == 3
    assert metrics.answerable_response_rate == metrics.answerable_rejection_rate == 0.5
    assert metrics.abstention_accuracy == metrics.safe_unanswerable_rate == 0.5
    assert metrics.unanswerable_false_answer_rate == 0.5
    assert metrics.rejection_rate == metrics.extractive_fallback_rate == 0.25
    assert metrics.status_counts == {"answered": 2, "abstained": 1, "rejected": 1}


def test_repeated_measurements_exclude_slow_warmup_and_preserve_zero_phases() -> None:
    service = FixtureService(slow_first=True)
    report = evaluate_rag(service, _queries(), repetitions=2, warmup_runs=1)
    assert len(service.calls) == 9
    assert report.warmup_runs == report.warmup_generation_count == 1
    assert report.dataset_query_count == 4
    assert report.metrics.query_count == 8
    assert [row.repetition for row in report.per_query] == [1] * 4 + [2] * 4
    assert report.metrics.mean_latency_ms == 45
    assert report.latency_by_stage["total"].p50_ms == 45
    assert report.latency_by_stage["total"].p95_ms == 80
    assert report.latency_by_stage["generation"].sample_count == 8
    assert report.latency_by_stage["generation"].min_ms == 0
    assert report.latency_by_stage["generation"].mean_ms == 31.25


def test_invalid_datasets_and_options_fail_before_service_calls() -> None:
    queries = _queries()
    for dataset in ([], queries[:2], queries[2:], queries + [queries[0]]):
        service = FixtureService()
        with pytest.raises(ValueError):
            evaluate_rag(service, dataset)
        assert not service.calls
    for repetitions, warmups in ((0, 0), (1, -1)):
        service = FixtureService()
        with pytest.raises(ValueError):
            evaluate_rag(service, queries, repetitions=repetitions, warmup_runs=warmups)
        assert not service.calls


def test_empty_or_inconsistent_labels_are_rejected() -> None:
    for values in (
        {"id": " ", "question": "valid", "answerable": False},
        {"id": "q", "question": " ", "answerable": False},
        {"id": "q", "question": "valid", "answerable": True},
        {"id": "q", "question": "valid", "answerable": False, "relevant_document_ids": ["a"]},
    ):
        with pytest.raises(ValidationError):
            GroundedEvaluationQuery.model_validate(values)


def test_report_roundtrip_and_markdown_keep_auditable_evidence(tmp_path: Path) -> None:
    dataset = tmp_path / "queries.jsonl"
    dataset.write_text("\n".join(query.model_dump_json() for query in _queries()))
    assert read_rag_queries(dataset) == _queries()
    report = evaluate_rag(FixtureService(), _queries(), metadata={"configuration": {"seed": 42}})
    output = tmp_path / "reports" / "rag.json"
    markdown = tmp_path / "docs" / "rag.md"
    write_rag_evaluation(output, report)
    _write_markdown(markdown, output)
    assert GroundedRAGEvaluation.model_validate_json(output.read_text()) == report
    assert json.loads(output.read_text())["per_query"][0]["question"] == "access"
    assert "q2" in markdown.read_text()
    assert "q4" in markdown.read_text()
    assert "case-insensitive substring" in markdown.read_text()
    assert "0 of those actually generated tokens" in markdown.read_text()


@pytest.mark.parametrize(
    "arguments",
    [
        ["--repetitions", "0"],
        ["--warmup-runs", "-1"],
        ["--seed", "4294967296"],
        ["--seed", "-1"],
    ],
)
def test_cli_rejects_invalid_counts_and_seed(arguments: list[str]) -> None:
    with pytest.raises(SystemExit):
        build_parser().parse_args(arguments)


def test_cli_rejects_colliding_output_paths_before_loading_inputs(tmp_path: Path) -> None:
    same = str(tmp_path / "same.json")
    with pytest.raises(SystemExit):
        main(["--output", same, "--markdown", same, "--queries", "missing.jsonl"])
