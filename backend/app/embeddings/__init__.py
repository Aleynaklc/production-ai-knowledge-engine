"""Dense embedding providers."""

from backend.app.embeddings.base import Embedder
from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbedder

__all__ = ["Embedder", "SentenceTransformerEmbedder"]
