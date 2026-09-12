"""Search the local NovaStack corpus with any Stage 5–11 retriever."""

import argparse
import json
from pathlib import Path

from qdrant_client import QdrantClient

from backend.app.config import Settings
from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from backend.app.ingestion.pipeline import read_chunks
from backend.app.retrieval.dense import QdrantDenseRetriever
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.models import Retriever
from backend.app.retrieval.reranker import CrossEncoderScorer, RerankedRetriever
from backend.app.retrieval.sparse import BM25Retriever

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Natural-language search query")
    parser.add_argument(
        "--method",
        choices=("dense", "bm25", "hybrid", "reranked"),
        default="reranked",
    )
    parser.add_argument("--top-k", type=int, default=None)
    return parser


def _print_results(retriever: Retriever, query: str, top_k: int) -> None:
    results = retriever.retrieve(query, top_k=top_k)
    records = [
        {
            "rank": result.rank,
            "score": round(result.score, 6),
            "chunk_id": result.chunk.chunk_id,
            "document_id": result.chunk.document_id,
            "source": result.chunk.source,
            "text": result.chunk.text,
            "component_scores": {
                name: round(score, 6) for name, score in result.component_scores.items()
            },
        }
        for result in results
    ]
    print(json.dumps(records, indent=2, ensure_ascii=False))


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()
    top_k = settings.retrieval_top_k if args.top_k is None else args.top_k
    if top_k < 1 or not args.query.strip():
        parser.error("query must contain text and top-k must be positive")
    chunks = read_chunks(PROJECT_ROOT / "data/processed/chunks_recursive.jsonl")
    bm25 = BM25Retriever(chunks)
    if args.method == "bm25":
        _print_results(bm25, args.query, top_k)
        return

    embedder = SentenceTransformerEmbedder(
        settings.embedding_model_name,
        settings.embedding_model_revision,
        settings.retrieval_device,
    )
    qdrant_path = settings.qdrant_path
    if not qdrant_path.is_absolute():
        qdrant_path = PROJECT_ROOT / qdrant_path
    client = QdrantClient(path=str(qdrant_path))
    try:
        dense = QdrantDenseRetriever(
            embedder,
            f"{settings.qdrant_collection}_recursive",
            client=client,
        )
        # A previous CLI/evaluation run may have indexed different chunks or a
        # different embedding model. Rebuild from the current input on each run.
        dense.index(chunks)
        if args.method == "dense":
            retriever: Retriever = dense
        else:
            hybrid = HybridRetriever(dense, bm25, settings.retrieval_candidate_k)
            if args.method == "hybrid":
                retriever = hybrid
            else:
                scorer = CrossEncoderScorer(
                    settings.reranker_model_name,
                    settings.reranker_model_revision,
                    settings.retrieval_device,
                )
                retriever = RerankedRetriever(
                    hybrid,
                    scorer,
                    settings.retrieval_candidate_k,
                )
        _print_results(retriever, args.query, top_k)
    finally:
        client.close()


if __name__ == "__main__":
    main()
