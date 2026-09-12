"""Persistent local Qdrant indexing and cosine retrieval."""

from collections.abc import Sequence
from pathlib import Path
from typing import Any, cast

from qdrant_client import QdrantClient, models

from backend.app.embeddings.base import Embedder
from backend.app.ingestion.models import DocumentChunk
from backend.app.retrieval.models import RetrievalResult


class QdrantDenseRetriever:
    """Index and search document chunks through a local Qdrant collection."""

    name = "dense"

    def __init__(
        self,
        embedder: Embedder,
        collection_name: str,
        *,
        path: Path | None = None,
        client: QdrantClient | None = None,
    ) -> None:
        if path is None and client is None:
            raise ValueError("Either a Qdrant path or an existing client is required")
        self.embedder = embedder
        self.collection_name = collection_name
        self._owns_client = client is None
        self.client = client or QdrantClient(path=str(path))

    def close(self) -> None:
        """Release local storage handles owned by this retriever."""

        if self._owns_client:
            self.client.close()

    def index(self, chunks: list[DocumentChunk], *, recreate: bool = True) -> None:
        """Create the collection and upsert all chunks with full source payloads."""

        if recreate and self.client.collection_exists(self.collection_name):
            self.client.delete_collection(self.collection_name)
        if not self.client.collection_exists(self.collection_name):
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=self.embedder.dimension,
                    distance=models.Distance.COSINE,
                ),
            )
        vectors = self.embedder.embed_documents([chunk.text for chunk in chunks])
        points = [
            models.PointStruct(
                id=chunk.chunk_id,
                vector=vector,
                payload=chunk.model_dump(mode="json"),
            )
            for chunk, vector in zip(chunks, vectors, strict=True)
        ]
        if points:
            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Embed a query and return Qdrant cosine matches."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        response = self.client.query_points(
            collection_name=self.collection_name,
            query=self.embedder.embed_query(query),
            limit=top_k,
            with_payload=True,
        )
        results: list[RetrievalResult] = []
        points = cast(Sequence[Any], response.points)
        for rank, point in enumerate(points, start=1):
            if point.payload is None:
                raise ValueError(f"Qdrant point {point.id} is missing its chunk payload")
            chunk = DocumentChunk.model_validate(point.payload)
            score = float(point.score)
            results.append(
                RetrievalResult(
                    chunk=chunk,
                    score=score,
                    rank=rank,
                    retriever=self.name,
                    component_scores={self.name: score},
                )
            )
        return results
