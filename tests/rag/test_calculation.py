"""Derived numbers are verified by bounded arithmetic rather than word overlap."""

import pytest

from backend.app.observability.tracing import build_system_trace
from backend.app.rag.calculation import table_total
from backend.app.rag.context import ContextBuilder
from backend.app.rag.service import RAGService
from tests.rag.test_rag import FakeGenerator, StaticRetriever, WordCodec, make_result


@pytest.mark.parametrize(
    ("quantity", "price", "expected"), [("4", "12", "48"), ("3", "7.25", "21.75"), ("0", "19", "0")]
)
def test_table_total_is_exact_and_never_calls_the_generator(
    quantity: str, price: str, expected: str
) -> None:
    evidence = make_result(
        1,
        f"| Product | Quantity | Unit price |\n|---|---|---|\n| Cedar bowl | {quantity} | {price} |",
    )
    generator = FakeGenerator("Invented answer")
    service = RAGService(
        StaticRetriever([evidence]),
        ContextBuilder(WordCodec(), token_budget=200),
        generator,
        attribute_sources=True,
    )
    answer = service.answer("What is the total value of all Cedar bowls in stock?")
    assert answer.status == "answered" and answer.calculation is not None
    assert answer.calculation.result == expected
    assert generator.calls == 0 and answer.raw_answer is None
    trace = build_system_trace(answer, "test")
    assert trace.stages[2].status == "skipped" and trace.stages[3].status == "completed"


@pytest.mark.parametrize(
    "question",
    [
        "What is the total value of Cedar bowls with tax?",
        "What is the price of Cedar bowls?",
        "What is the total value of missing items?",
    ],
)
def test_calculation_does_not_infer_unknown_operations_or_items(question: str) -> None:
    context = ContextBuilder(WordCodec(), token_budget=200).build(
        question,
        [
            make_result(
                1, "| Product | Quantity | Unit price |\n|---|---|---|\n| Cedar bowl | 3 | 7 |"
            )
        ],
    )
    assert table_total(question, context) is None


def test_conflicting_rows_are_not_silently_selected() -> None:
    context = ContextBuilder(WordCodec(), token_budget=300).build(
        "question",
        [
            make_result(
                1,
                "| Product | Quantity | Unit price |\n|---|---|---|\n| Cedar bowl | 3 | 7 |\n| Cedar bowl | 3 | 8 |",
            )
        ],
    )
    assert table_total("What is the total value of Cedar bowls?", context) is None
