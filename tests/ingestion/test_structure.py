"""Preserve source information while enforcing actual token and structural bounds."""

from backend.app.ingestion.chunkers import FixedTokenChunker, RecursiveChunker
from tests.ingestion.test_ingestion import WordCodec


def test_oversized_prose_keeps_overlap_and_every_word() -> None:
    words = [f"word{index}" for index in range(120)]
    chunker = RecursiveChunker(WordCodec(), 20, 5)
    chunks = chunker.chunk(" ".join(words))
    assert all(chunker.codec.count(chunk) <= 20 for chunk in chunks)
    assert set(" ".join(chunks).split()) == set(words)
    for one, two in zip(chunks, chunks[1:], strict=False):
        assert one.split()[-5:] == two.split()[:5]


def test_fenced_code_keeps_blank_lines_and_indentation_when_it_fits() -> None:
    code = "```python\ndef run():\n    a = 1\n\n    return a\n```"
    assert RecursiveChunker(WordCodec(), 30, 3).chunk(code) == [code]


def test_long_code_fragments_have_fences_and_bounded_lengths() -> None:
    body = "\n".join(f"value_{index} = {index}" for index in range(40))
    chunks = RecursiveChunker(WordCodec(), 30, 3).chunk(f"```python\n{body}\n```")
    assert len(chunks) > 1
    assert all(chunk.startswith("```python\n") and chunk.endswith("\n```") for chunk in chunks)
    assert all(WordCodec().count(chunk) <= 30 for chunk in chunks)
    assert all(f"value_{index}" in " ".join(chunks) for index in range(40))


def test_table_fragments_repeat_column_names_without_losing_rows() -> None:
    header = "| Item | Quantity |\n|---|---|"
    rows = [f"| item{index} | {index} |" for index in range(20)]
    chunks = RecursiveChunker(WordCodec(), 30, 3).chunk(header + "\n" + "\n".join(rows))
    assert len(chunks) > 1
    assert all(chunk.startswith(header) for chunk in chunks)
    assert all(WordCodec().count(chunk) <= 30 for chunk in chunks)
    assert all(row in "\n".join(chunks) for row in rows)


def test_fixed_windows_check_retokenized_length() -> None:
    class BoundaryCodec(WordCodec):
        def count(self, text: str) -> int:
            return super().count(text) + 1

    chunks = FixedTokenChunker(BoundaryCodec(), 5, 1).chunk("one two three four five six seven")
    assert all(BoundaryCodec().count(chunk) <= 5 for chunk in chunks)
    assert set(" ".join(chunks).split()) == set("one two three four five six seven".split())
