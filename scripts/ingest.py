"""Ingest the local corpus with fixed and recursive token chunking."""

import argparse
import json
from pathlib import Path

from backend.app.config import Settings
from backend.app.evaluation.dataset import (
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
from backend.app.ingestion.pipeline import ingest_directory

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

    args = build_parser().parse_args()
    settings = Settings()
    input_directory = args.input_directory.resolve()
    chunk_size = args.chunk_size or settings.chunk_size_tokens
    overlap = settings.chunk_overlap_tokens if args.overlap is None else args.overlap
    codec = HuggingFaceTokenCodec.from_pretrained(
        settings.model_name,
        settings.model_revision,
    )
    source_path = PROJECT_ROOT / "data/evaluation/retrieval_queries_source.jsonl"
    raw_queries = read_queries(source_path, SourceEvaluationQuery)
    if not all(isinstance(query, SourceEvaluationQuery) for query in raw_queries):
        raise TypeError("Source query file returned an unexpected query type")
    source_queries = list(raw_queries)
    strategies = ("fixed", "recursive") if args.strategy == "all" else (args.strategy,)

    summaries: dict[str, object] = {}
    for strategy in strategies:
        output_path = PROJECT_ROOT / f"data/processed/chunks_{strategy}.jsonl"
        chunks, summary = ingest_directory(
            input_directory,
            output_path,
            _chunker(strategy, codec, chunk_size, overlap),
        )
        resolved = resolve_queries(source_queries, chunks)
        write_queries(
            PROJECT_ROOT / f"data/evaluation/retrieval_queries_{strategy}.jsonl",
            resolved,
        )
        summaries[strategy] = summary.model_dump(mode="json")

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
