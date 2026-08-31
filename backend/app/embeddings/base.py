"""Stable embedding boundary used by indexing and retrieval."""

from typing import Protocol


class Embedder(Protocol):
    """Convert text into equal-sized dense vectors."""

    @property
    def dimension(self) -> int:
        """Return the output vector width."""

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed a batch of source passages."""

    def embed_query(self, text: str) -> list[float]:
        """Embed one search query in the same vector space."""
