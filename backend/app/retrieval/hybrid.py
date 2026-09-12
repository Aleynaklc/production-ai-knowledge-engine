"""Dense plus lexical retrieval joined through rank fusion."""

from backend.app.retrieval.fusion import reciprocal_rank_fusion
from backend.app.retrieval.models import RetrievalResult, Retriever


class HybridRetriever:
    """Retrieve candidates independently, then combine their ranks."""

    name = "hybrid_rrf"

    def __init__(self, dense: Retriever, sparse: Retriever, candidate_k: int = 20) -> None:
        if candidate_k < 1:
            raise ValueError("candidate_k must be positive")
        self.dense = dense
        self.sparse = sparse
        self.candidate_k = candidate_k

    def retrieve(self, query: str, top_k: int = 5) -> list[RetrievalResult]:
        """Fuse dense and BM25 candidates with RRF."""

        if top_k < 1:
            raise ValueError("top_k must be positive")
        return reciprocal_rank_fusion(
            [
                self.dense.retrieve(query, self.candidate_k),
                self.sparse.retrieve(query, self.candidate_k),
            ],
            top_k=top_k,
        )
