"""Ingest the local corpus with fixed and recursive token chunking."""

import argparse
import json
from pathlib import Path

from backend.app.config import Settings
from backend.app.evaluation.dataset import (
    RetrievalEvaluationQuery,
    SourceEvaluationQuery,
    read_queries,
    resolve_queries,
    write_queries,
)
from backend.app.ingestion.chunkers import (
    BaseChunker,
    FixedTokenChunker,
    HuggingFaceTokenCodec,
    RecursiveChunker,
)
from backend.app.ingestion.models import DocumentChunk
from backend.app.ingestion.parser import load_documents
from backend.app.ingestion.pipeline import build_chunks, summarize, write_chunks

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _display_path(path: Path) -> str:
    try:
        return path.relative_to(PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def build_parser() -> argparse.ArgumentParser:
    """Define the ingestion command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_directory", type=Path, nargs="?", default=Path("data/raw"))
    parser.add_argument(
        "--strategy",
        choices=("all", "fixed", "recursive"),
        default="all",
        help="Chunking strategy to persist; all writes both comparison datasets.",
    )
    parser.add_argument("--chunk-size", type=int, default=None)
    parser.add_argument("--overlap", type=int, default=None)
    return parser


def _chunker(
    strategy: str,
    codec: HuggingFaceTokenCodec,
    chunk_size: int,
    overlap: int,
) -> BaseChunker:
    chunker_type = FixedTokenChunker if strategy == "fixed" else RecursiveChunker
    return chunker_type(codec, chunk_size, overlap)


def main() -> None:
    """Load documents, persist chunks, labels, and ingestion statistics."""

    parser = build_parser()
    args = parser.parse_args()
    settings = Settings()
    input_directory = args.input_directory.resolve()
    chunk_size = settings.chunk_size_tokens if args.chunk_size is None else args.chunk_size
    overlap = settings.chunk_overlap_tokens if args.overlap is None else args.overlap
    if not input_directory.is_dir():
        parser.error("input_directory must be an existing directory")
    if chunk_size <= 0 or not 0 <= overlap < chunk_size:
        parser.error("require chunk-size > 0 and 0 <= overlap < chunk-size")
    documents = load_documents(input_directory)
    if not documents:
        parser.error("input_directory contains no supported documents")
    codec = HuggingFaceTokenCodec.from_pretrained(
        settings.model_name,
        settings.model_revision,
    )
    source_path = PROJECT_ROOT / "data/evaluation/retrieval_queries_source.jsonl"
    raw_queries = read_queries(source_path, SourceEvaluationQuery)
    strategies = ("fixed", "recursive") if args.strategy == "all" else (args.strategy,)

    summaries: dict[str, object] = {}
    prepared: dict[str, tuple[list[DocumentChunk], list[RetrievalEvaluationQuery]]] = {}
    for strategy in strategies:
        chunker = _chunker(strategy, codec, chunk_size, overlap)
        chunks = build_chunks(documents, chunker)
        # Validate every strategy before changing any existing corpus or label files.
        resolved = resolve_queries(raw_queries, chunks)
        prepared[strategy] = (chunks, resolved)
        summaries[strategy] = summarize(chunks, documents, chunker).model_dump(mode="json")

    for strategy, (chunks, resolved) in prepared.items():
        write_chunks(PROJECT_ROOT / f"data/processed/chunks_{strategy}.jsonl", chunks)
        write_queries(
            PROJECT_ROOT / f"data/evaluation/retrieval_queries_{strategy}.jsonl",
            resolved,
        )

    report_path = PROJECT_ROOT / "evaluation/reports/ingestion_v1.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(
        json.dumps(
            {
                "schema_version": 1,
                "source_directory": _display_path(input_directory),
                "tokenizer": {
                    "model_name": settings.model_name,
                    "revision": settings.model_revision,
                },
                "strategies": summaries,
            },
            indent=2,
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summaries, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
