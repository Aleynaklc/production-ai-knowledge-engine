"""Sentence Transformers implementation with normalized cosine vectors."""

from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer

from backend.app.llm.model import DeviceRequest, resolve_device


class SentenceTransformerEmbedder:
    """Load one pinned Sentence Transformer and expose plain Python vectors."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        requested_device: DeviceRequest = "auto",
    ) -> None:
        self.model_name = model_name
        self.revision = revision
        self.device = resolve_device(requested_device)
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device.type,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("Embedding model did not declare an output dimension")
        self._dimension = int(dimension)

    @property
    def dimension(self) -> int:
        """Return the model's sentence embedding width."""

        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=32,
            convert_to_numpy=True,
            normalize_embeddings=True,
            show_progress_bar=False,
        )
        array = np.asarray(vectors, dtype=np.float32)
        return cast(list[list[float]], array.tolist())

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        """Embed passages in deterministic input order."""

        return self._encode(texts)

    def embed_query(self, text: str) -> list[float]:
        """Embed one query."""

        return self._encode([text])[0]
