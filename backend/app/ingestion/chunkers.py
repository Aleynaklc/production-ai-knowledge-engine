"""Transparent fixed-token and structure-aware recursive chunkers."""

import re
from abc import ABC, abstractmethod
from collections.abc import Sequence
from typing import Protocol, cast

from transformers import AutoTokenizer, PreTrainedTokenizerBase


class TokenCodec(Protocol):
    """Minimal token operations required by chunking strategies."""

    def count(self, text: str) -> int:
        """Return the number of model tokens in text."""

    def offsets(self, text: str) -> list[tuple[int, int]]:
        """Return character offsets for every token in text."""


class HuggingFaceTokenCodec:
    """Fast-tokenizer adapter used by production ingestion."""

    def __init__(self, tokenizer: PreTrainedTokenizerBase) -> None:
        self._tokenizer = tokenizer

    @classmethod
    def from_pretrained(cls, model_name: str, revision: str) -> "HuggingFaceTokenCodec":
        tokenizer = cast(
            PreTrainedTokenizerBase,
            AutoTokenizer.from_pretrained(model_name, revision=revision, use_fast=True),
        )
        if not tokenizer.is_fast:
            raise ValueError("Chunking requires a fast tokenizer with offset mappings")
        return cls(tokenizer)

    def count(self, text: str) -> int:
        return len(self._tokenizer.encode(text, add_special_tokens=False))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        encoded = self._tokenizer(
            text,
            add_special_tokens=False,
            return_attention_mask=False,
            return_offsets_mapping=True,
        )
        raw_offsets = cast(Sequence[Sequence[int]], encoded["offset_mapping"])
        return [(int(offset[0]), int(offset[1])) for offset in raw_offsets]


class SharedTokenBudget:
    """Use generator offsets but enforce both generator and encoder token counts."""

    def __init__(self, generator: TokenCodec, encoder: TokenCodec) -> None:
        self.generator, self.encoder = generator, encoder

    def count(self, text: str) -> int:
        return max(self.generator.count(text), self.encoder.count(text))

    def offsets(self, text: str) -> list[tuple[int, int]]:
        return self.generator.offsets(text)


class BaseChunker(ABC):
    """Common interface for configurable document chunking."""

    name: str

    def __init__(self, codec: TokenCodec, chunk_size: int, overlap: int) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be positive")
        if overlap < 0 or overlap >= chunk_size:
            raise ValueError("overlap must satisfy 0 <= overlap < chunk_size")
        self.codec = codec
        self.chunk_size = chunk_size
        self.overlap = overlap

    @abstractmethod
    def chunk(self, text: str) -> list[str]:
        """Split text into ordered, non-empty passages."""


class FixedTokenChunker(BaseChunker):
    """Slice the original text with fixed token windows and token overlap."""

    name = "fixed"

    def chunk(self, text: str) -> list[str]:
        offsets = self.codec.offsets(text)
        if not offsets:
            return []
        chunks: list[str] = []
        start = 0
        while start < len(offsets):
            end = min(start + self.chunk_size, len(offsets))
            passage = text[offsets[start][0] : offsets[end - 1][1]].strip()
            # Slicing and retokenizing can change BPE boundaries. Enforce the
            # actual emitted length, not just a count of original offsets.
            while self.codec.count(passage) > self.chunk_size and end > start + 1:
                end -= 1
                passage = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if self.codec.count(passage) > self.chunk_size:
                raise ValueError("A token span cannot fit the configured chunk size")
            if passage:
                chunks.append(passage)
            if end == len(offsets):
                break
            start = max(start + 1, end - self.overlap)
        return chunks


class RecursiveChunker(BaseChunker):
    """Prefer Markdown blocks and sentences before falling back to token windows."""

    name = "recursive"

    def _split_oversized(self, block: str) -> list[str]:
        if self.codec.count(block) <= self.chunk_size:
            return [block]
        lines = block.splitlines()
        if (
            len(lines) > 2
            and lines[0].lstrip().startswith("|")
            and re.fullmatch(r"[\s|:\-]+", lines[1])
        ):
            header = "\n".join(lines[:2])
            reserve = self.codec.count(header) + 4
            if reserve + self.overlap < self.chunk_size:
                chunks: list[str] = []
                current = header
                for row in lines[2:]:
                    candidate = current + "\n" + row
                    if self.codec.count(candidate) <= self.chunk_size:
                        current = candidate
                        continue
                    if current != header:
                        chunks.append(current)
                    parts = FixedTokenChunker(
                        self.codec, self.chunk_size - reserve, self.overlap
                    ).chunk(row)
                    chunks.extend(header + "\n" + part for part in parts[:-1])
                    current = header + "\n" + parts[-1]
                if current != header:
                    chunks.append(current)
                if all(self.codec.count(part) <= self.chunk_size for part in chunks):
                    return chunks
        opening = re.match(r"^(`{3,}|~{3,})(.*)$", lines[0])
        if opening and len(lines) > 2 and lines[-1].strip() == opening[1]:
            prefix, suffix = lines[0] + "\n", "\n" + opening[1]
            # Long code fragments retain explicit fences and overlap. They are
            # fragments, not promises of independently executable functions.
            reserve = self.codec.count(prefix + suffix) + 4
            size = self.chunk_size - reserve
            if size > self.overlap:
                body = "\n".join(lines[1:-1])
                while size > self.overlap:
                    parts = FixedTokenChunker(self.codec, size, self.overlap).chunk(body)
                    wrapped = [prefix + part + suffix for part in parts]
                    if all(self.codec.count(part) <= self.chunk_size for part in wrapped):
                        return wrapped
                    size -= 1
        return FixedTokenChunker(self.codec, self.chunk_size, self.overlap).chunk(block)

    def _tail(self, text: str) -> str:
        if self.overlap == 0 or "```" in text or "~~~" in text:
            return ""
        offsets = self.codec.offsets(text)
        if not offsets:
            return ""
        return text[offsets[max(0, len(offsets) - self.overlap)][0] :].strip()

    @staticmethod
    def _blocks(text: str) -> list[str]:
        # Blank lines inside fenced code are not paragraph boundaries.
        blocks: list[str] = []
        current: list[str] = []
        fence = ""
        for line in text.splitlines():
            marker = re.match(r"^ {0,3}(`{3,}|~{3,})", line)
            if marker:
                if not fence:
                    if current:
                        blocks.append("\n".join(current).strip())
                        current = []
                    fence = marker[1]
                elif marker[1] == fence:
                    current.append(line)
                    blocks.append("\n".join(current).strip())
                    current = []
                    fence = ""
                    continue
            if not fence and not line.strip():
                if current:
                    blocks.append("\n".join(current).strip())
                    current = []
            else:
                current.append(line)
        if current:
            blocks.append("\n".join(current).strip())
        return [block for block in blocks if block]

    def chunk(self, text: str) -> list[str]:
        chunks: list[str] = []
        current = ""
        for block in self._blocks(text):
            candidate = f"{current}\n\n{block}".strip()
            if self.codec.count(candidate) <= self.chunk_size:
                current = candidate
                continue
            if current:
                chunks.append(current)
            overlap_text = self._tail(current)
            candidate = f"{overlap_text}\n\n{block}".strip()
            if self.codec.count(candidate) <= self.chunk_size:
                current = candidate
            elif self.codec.count(block) > self.chunk_size:
                parts = self._split_oversized(block)
                chunks.extend(parts[:-1])
                current = parts[-1]
            elif block.startswith(("```", "~~~", "|")):
                # A complete code/table block takes precedence over copying a
                # prose tail. Context expansion reconnects adjacent passages.
                current = block
            else:
                parts = FixedTokenChunker(self.codec, self.chunk_size, self.overlap).chunk(
                    candidate
                )
                chunks.extend(parts[:-1])
                current = parts[-1]
        if current:
            chunks.append(current)
        return chunks
