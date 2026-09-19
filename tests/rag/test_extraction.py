"""Only explicit structured requests use source excerpts, with source provenance."""

import pytest

from backend.app.rag.context import ContextBuilder
from backend.app.rag.extraction import extract_requested_evidence
from backend.app.rag.service import RAGService
from tests.rag.test_rag import FakeGenerator, StaticRetriever, WordCodec, make_result


@pytest.mark.parametrize("name", ["Aster", "Birch", "Cobalt"])
def test_code_examples_copy_actual_blocks_instead_of_generating_code(name: str) -> None:
    code = "```python\ndef double(n):\n    return n * 2\n```"
    source = make_result(1, f"# {name} examples\n\n## Doubling\n{code}")
    generator = FakeGenerator("Untrusted invented code")
    service = RAGService(
        StaticRetriever([source]),
        ContextBuilder(WordCodec(), token_budget=300),
        generator,
        attribute_sources=True,
    )
    result = service.answer(f"Give examples from {name}")
    assert result.extraction is not None and result.status == "answered"
    assert code in result.answer and result.citations[0].chunk_id == source.chunk.chunk_id
    assert generator.calls == 0


def test_procedure_preserves_every_step_and_rejects_new_conditions() -> None:
    text = "# Atlas restart\n1. Stop requests.\n2. Drain requests.\n3. Restart Atlas.\n4. Check health."
    context = ContextBuilder(WordCodec(), token_budget=300).build(
        "question", [make_result(1, text)]
    )
    result = extract_requested_evidence("How should Atlas be restarted?", context)
    assert result is not None and len(result.excerpts) == 4
    assert all(excerpt.text in text for excerpt in result.excerpts)
    assert (
        extract_requested_evidence("How should Atlas be restarted without downtime?", context)
        is None
    )


def test_example_extraction_does_not_pretend_to_explain_or_modify_code() -> None:
    context = ContextBuilder(WordCodec(), token_budget=300).build(
        "question", [make_result(1, "```python\nprint(1)\n```")]
    )
    assert extract_requested_evidence("Explain this example", context) is None
    assert extract_requested_evidence("Give examples without print", context) is None
    assert extract_requested_evidence("Give examples of database transactions", context) is None
