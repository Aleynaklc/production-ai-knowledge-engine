"""Prevent permissive metrics from hiding wrong numbers, missing sources, or false answers."""

from pathlib import Path

from backend.app.evaluation.workspace import contains_term, read_scenarios, score_answer, summarize
from tests.test_api_tracing import FakeAnswerService


def test_cross_domain_dataset_has_positive_and_negative_coverage() -> None:
    dataset = read_scenarios(Path("data/evaluation/workspace_scenarios.json"))
    assert len(dataset.scenarios) >= 8
    assert sum(len(scenario.queries) for scenario in dataset.scenarios) >= 40
    assert (
        len({query.category for scenario in dataset.scenarios for query in scenario.queries}) >= 12
    )
    for scenario in dataset.scenarios:
        assert any(query.answerable for query in scenario.queries)
        assert any(not query.answerable for query in scenario.queries)


def test_quality_gate_rejects_wrong_answers_even_if_citation_validator_accepts_them() -> None:
    dataset = read_scenarios(Path("data/evaluation/workspace_scenarios.json"))
    positive, negative = dataset.scenarios[0].queries[0], dataset.scenarios[0].queries[-1]
    answer = FakeAnswerService().answer("test")
    positive_row = score_answer(positive, answer, {})
    negative_row = score_answer(negative, answer, {})
    summary = summarize([positive_row, negative_row])
    assert answer.validation.valid  # Internal validation alone does not establish correctness.
    assert not positive_row["passed_lexical_regression"]
    assert summary["unanswerable_false_answers"] == 1
    assert not summary["quality_gate_passed"]
    assert not contains_term("The period is 40 days", "4")
    assert not contains_term("Mina", "Min")
    assert contains_term("The period is 40 days", "40")


def test_document_recall_is_not_a_substitute_for_actual_evidence_spans() -> None:
    from backend.app.evaluation.workspace import ScenarioQuery, score_answer
    from backend.app.rag.context import ContextBuilder
    from backend.app.rag.service import RAGService
    from tests.rag.test_rag import FakeGenerator, StaticRetriever, WordCodec, make_result

    query = ScenarioQuery(
        id="span",
        question="What is the limit?",
        category="fact",
        answerable=True,
        relevant_document_ids=["doc-1"],
        required_answer_terms=["30"],
        required_evidence_spans=["The limit is 30."],
    )
    answer = RAGService(
        StaticRetriever([make_result(1, "The limit is 30.")]),
        ContextBuilder(WordCodec(), token_budget=100),
        FakeGenerator("[S1] The limit is 30."),
    ).answer(query.question)
    result = score_answer(query, answer, {}, context_text="Unrelated text from the same document.")
    assert result["retrieval_recall"] == 1.0
    assert result["evidence_span_recall"] == 0.0


def test_acceptance_labels_are_frozen_and_disjoint_from_development() -> None:
    import hashlib
    import json

    path = Path("data/evaluation/workspace_acceptance.json")
    manifest = json.loads(Path("data/evaluation/workspace_acceptance_manifest.json").read_text())
    assert hashlib.sha256(path.read_bytes()).hexdigest() == manifest["dataset_sha256"]
    development = read_scenarios(Path("data/evaluation/workspace_scenarios.json"))
    acceptance = read_scenarios(path)
    dev_questions = {q.question for s in development.scenarios for q in s.queries}
    assert not dev_questions & {q.question for s in acceptance.scenarios for q in s.queries}
    dev_sources = {d.text for s in development.scenarios for d in s.documents}
    assert not dev_sources & {d.text for s in acceptance.scenarios for d in s.documents}


def test_evidence_labels_must_exist_in_the_labeled_document() -> None:
    import pytest
    from pydantic import ValidationError

    dataset = read_scenarios(Path("data/evaluation/workspace_acceptance.json"))
    scenario = dataset.scenarios[0].model_dump()
    scenario["queries"][0]["required_evidence_spans"] = ["This fact is not in the source"]
    with pytest.raises(ValidationError, match="Evidence span is absent"):
        type(dataset.scenarios[0]).model_validate(scenario)
