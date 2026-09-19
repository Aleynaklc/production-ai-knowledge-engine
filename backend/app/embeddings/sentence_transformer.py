"""Sentence Transformers implementation with normalized cosine vectors."""

from typing import cast

import numpy as np
from sentence_transformers import SentenceTransformer
from transformers import PreTrainedTokenizerBase

from backend.app.llm.model import DeviceRequest, resolve_device


class SentenceTransformerEmbedder:
    """Load one pinned Sentence Transformer and expose plain Python vectors."""

    def __init__(
        self,
        model_name: str,
        revision: str,
        requested_device: DeviceRequest = "auto",
        *,
        batch_size: int = 32,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("Embedding batch_size must be positive")
        self.model_name = model_name
        self.revision = revision
        self.device = resolve_device(requested_device)
        self.batch_size = batch_size
        self._model = SentenceTransformer(
            model_name,
            revision=revision,
            device=self.device.type,
        )
        dimension = self._model.get_embedding_dimension()
        if dimension is None:
            raise ValueError("Embedding model did not declare an output dimension")
        self._dimension = int(dimension)

    def metadata(self) -> dict[str, str | int]:
        """Describe the pinned encoder and raw parameter footprint (not peak RAM)."""

        return {
            "model_name": self.model_name,
            "revision": self.revision,
            "device": self.device.type,
            "dimension": self.dimension,
            "batch_size": self.batch_size,
            "max_sequence_length": int(self._model.max_seq_length),
            "parameter_count": sum(parameter.numel() for parameter in self._model.parameters()),
            "parameter_bytes": sum(
                parameter.numel() * parameter.element_size()
                for parameter in self._model.parameters()
            ),
        }

    @property
    def tokenizer(self) -> PreTrainedTokenizerBase:
        return cast(PreTrainedTokenizerBase, self._model.tokenizer)

    @property
    def max_content_tokens(self) -> int:
        return int(self._model.max_seq_length) - self.tokenizer.num_special_tokens_to_add(
            pair=False
        )

    @property
    def dimension(self) -> int:
        """Return the model's sentence embedding width."""

        return self._dimension

    def _encode(self, texts: list[str]) -> list[list[float]]:
        vectors = self._model.encode(
            texts,
            batch_size=self.batch_size,
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
