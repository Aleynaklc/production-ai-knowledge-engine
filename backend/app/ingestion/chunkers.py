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
        stride = self.chunk_size - self.overlap
        chunks: list[str] = []
        for start in range(0, len(offsets), stride):
            end = min(start + self.chunk_size, len(offsets))
            passage = text[offsets[start][0] : offsets[end - 1][1]].strip()
            if passage:
                chunks.append(passage)
            if end == len(offsets):
                break
        return chunks


class RecursiveChunker(BaseChunker):
    """Prefer Markdown blocks and sentences before falling back to token windows."""

    name = "recursive"

    def _split_oversized(self, block: str) -> list[str]:
        if self.codec.count(block) <= self.chunk_size:
            return [block]
        sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", block) if part.strip()]
        if len(sentences) > 1:
            units: list[str] = []
            current = ""
            for sentence in sentences:
                candidate = f"{current} {sentence}".strip()
                if current and self.codec.count(candidate) > self.chunk_size:
                    units.append(current)
                    current = sentence
                else:
                    current = candidate
            if current:
                units.append(current)
            expanded: list[str] = []
            fixed = FixedTokenChunker(self.codec, self.chunk_size, 0)
            for unit in units:
                expanded.extend(
                    fixed.chunk(unit) if self.codec.count(unit) > self.chunk_size else [unit]
                )
            return expanded
        return FixedTokenChunker(self.codec, self.chunk_size, 0).chunk(block)

    def _tail(self, text: str) -> str:
        if self.overlap == 0:
            return ""
        offsets = self.codec.offsets(text)
        if not offsets:
            return ""
        start = offsets[max(0, len(offsets) - self.overlap)][0]
        return text[start:].strip()

    def chunk(self, text: str) -> list[str]:
        blocks = [block.strip() for block in re.split(r"\n{2,}", text) if block.strip()]
        units = [unit for block in blocks for unit in self._split_oversized(block)]
        chunks: list[str] = []
        current = ""
        for unit in units:
            candidate = f"{current}\n\n{unit}".strip()
            if not current or self.codec.count(candidate) <= self.chunk_size:
                current = candidate
                continue
            chunks.append(current)
            overlap_text = self._tail(current)
            candidate = f"{overlap_text}\n\n{unit}".strip()
            current = candidate if self.codec.count(candidate) <= self.chunk_size else unit
        if current:
            chunks.append(current)
        return chunks
