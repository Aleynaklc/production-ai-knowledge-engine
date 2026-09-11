"""Stage 24: compare chunk windows with one pinned embedding model and fresh labels."""

import argparse
import json
import platform
import shlex
import sys
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

from backend.app.config import Settings
from backend.app.embeddings.sentence_transformer import SentenceTransformerEmbedder
from backend.app.evaluation.chunking import (
    ChunkingBenchmarkConfig,
    ChunkingVariant,
    benchmark_chunking,
    chunking_markdown,
)
from backend.app.evaluation.dataset import SourceEvaluationQuery, read_queries
from backend.app.ingestion.chunkers import HuggingFaceTokenCodec
from backend.app.ingestion.parser import load_documents
from backend.app.llm.generation import _synchronize

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def build_parser() -> argparse.ArgumentParser:
    """Expose explicit experiment settings; no application collection is opened."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--corpus", type=Path, default=PROJECT_ROOT / "data/raw")
    parser.add_argument(
        "--queries",
        type=Path,
        default=PROJECT_ROOT / "data/evaluation/retrieval_queries_source.jsonl",
    )
    parser.add_argument(
        "--strategies", nargs="+", choices=("fixed", "recursive"), default=["fixed", "recursive"]
    )
    parser.add_argument("--sizes", type=int, nargs="+", default=[80, 160, 240])
    parser.add_argument("--overlaps", type=int, nargs="+", default=[0, 30])
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--repetitions", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--device", choices=("auto", "cpu", "mps", "cuda"), default=None)
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "evaluation/reports/chunking_benchmark_v1.json",
    )
    parser.add_argument(
        "--markdown", type=Path, default=PROJECT_ROOT / "docs/chunking-benchmark.md"
    )
    return parser


def main() -> None:
    """Load real pinned models once, measure all variants, and write JSON/Markdown evidence."""

    parser = build_parser()
    args = parser.parse_args()
    try:
        config = ChunkingBenchmarkConfig(
            top_k=args.top_k,
            repetitions=args.repetitions,
            warmup_runs=args.warmup_runs,
        )
        variants = [
            ChunkingVariant(strategy=strategy, chunk_size_tokens=size, overlap_tokens=overlap)
            for strategy in args.strategies
            for size in args.sizes
            for overlap in args.overlaps
        ]
        if len({variant.name for variant in variants}) != len(variants):
            raise ValueError("Do not repeat strategies, sizes, or overlaps")
        if not args.corpus.is_dir():
            raise ValueError(f"Corpus directory does not exist: {args.corpus}")
        if args.output.resolve() == args.markdown.resolve():
            raise ValueError("JSON and Markdown output paths must differ")
        started = perf_counter()
        documents = load_documents(args.corpus)
        queries = read_queries(args.queries, SourceEvaluationQuery)
        input_load_ms = (perf_counter() - started) * 1_000
        if not documents or not queries or not any(q.category != "unanswerable" for q in queries):
            raise ValueError("Supply a non-empty corpus and at least one answerable query")
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    settings = Settings()
    started = perf_counter()
    codec = HuggingFaceTokenCodec.from_pretrained(settings.model_name, settings.model_revision)
    tokenizer_load_ms = (perf_counter() - started) * 1_000
    started = perf_counter()
    embedder = SentenceTransformerEmbedder(
        settings.embedding_model_name,
        settings.embedding_model_revision,
        args.device or settings.retrieval_device,
    )
    _synchronize(embedder.device)
    embedding_load_ms = (perf_counter() - started) * 1_000
    # This public ST tokenizer reports special tokens too, matching model input truncation.
    embedding_tokenizer = embedder._model.tokenizer
    embedding_max_tokens = embedder._model.max_seq_length
    result = benchmark_chunking(
        documents,
        queries,
        codec,
        embedder,
        variants,
        config=config,
        embedding_token_count=lambda text: len(
            embedding_tokenizer.encode(
                text, add_special_tokens=True, truncation=False, verbose=False
            )
        ),
        embedding_max_tokens=embedding_max_tokens,
    )
    command = shlex.join([sys.executable, "-m", "scripts.benchmark_chunking", *sys.argv[1:]])
    metadata = {
        "schema_version": 1,
        "stage": 24,
        "generated_at": datetime.now(UTC).isoformat(),
        "command": command,
        "inputs": {"corpus": str(args.corpus.resolve()), "queries": str(args.queries.resolve())},
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "processor": platform.processor(),
            "packages": {
                name: version(name)
                for name in (
                    "numpy",
                    "qdrant-client",
                    "sentence-transformers",
                    "transformers",
                    "torch",
                )
            },
        },
        "models": {
            "tokenizer": {"name": settings.model_name, "revision": settings.model_revision},
            "embedding": {
                "name": settings.embedding_model_name,
                "revision": settings.embedding_model_revision,
                "dimension": embedder.dimension,
                "device": str(embedder.device),
                "max_sequence_tokens": embedding_max_tokens,
                "normalize_embeddings": True,
                "batch_size": 32,
            },
        },
        "retrieval": {"backend": "qdrant_in_memory", "distance": "cosine", "reranker": None},
        "shared_setup_ms": {
            "input_load": round(input_load_ms, 6),
            "tokenizer_load": round(tokenizer_load_ms, 6),
            "embedding_model_load": round(embedding_load_ms, 6),
        },
        "benchmark": result.model_dump(mode="json"),
    }
    table = chunking_markdown(result)
    markdown = f"""# Stage 24 — Chunking Benchmark

Generated from a real model run at {metadata["generated_at"]}.

```sh
{command}
```

{table}

## Reproducibility and controls

- Embeddings: `{settings.embedding_model_name}` revision `{settings.embedding_model_revision}`;
  dimension {embedder.dimension}, device `{embedder.device}`, normalized vectors, batch size 32.
- Token counts: `{settings.model_name}` revision `{settings.model_revision}`.
- Corpus: {result.document_count} documents, {result.source_token_count} tokenizer tokens;
  SHA-256 `{result.source_sha256}`.
- Source labels SHA-256: `{result.labels_sha256}`.
- Retrieval: cosine dense search in separate in-memory Qdrant indexes, top {config.top_k};
  {config.repetitions} measured query passes after {config.warmup_runs} warmup queries per index.
- JSON evidence: `{args.output}`; includes source-resolved labels, per-query ranked IDs,
  Precision/NDCG and other cutoffs, all trial latencies, token distributions, build timings,
  Python/package/platform versions, and embedding truncation counts.

## Measurement boundaries

Ingestion is chunking plus embedding/index creation, measured once per variant. Index time
includes document embedding and Qdrant payload construction/upsert; JSON also separates
embedding and non-embedding index time. Shared input/model/tokenizer loading is measured
separately and excluded. Query latency includes query embedding and dense search, excludes
warmup, and pools every measured query pass, including unanswerable queries. Wall-clock
timings depend on hardware, system load, model caches, and variant execution order; these
are local comparisons, not a production Qdrant throughput or memory benchmark.

Chunk token expansion is total chunk tokens divided by original source tokens. It includes
overlap duplication and tokenizer boundary effects. Raw float32 vector bytes and UTF-8
chunk text bytes are size estimates, not measured process memory or full index storage.
Embedding truncation counts use the embedding tokenizer including special tokens; chunk
sizes use the ingestion tokenizer, so their lengths can differ. Recursive overlap is a
requested maximum: the chunker drops overlap when a new unit would exceed its token limit.

## Quality interpretation

Source anchors are resolved afresh for each variant to one representative chunk per anchor
(exact substring first, otherwise at least 60% anchor-token coverage). Overlapping chunks
may contain equivalent evidence without receiving a gold label; changing windows can
also change the number of distinct gold chunks. This label policy can bias comparisons.
Recall is retrieved relevant chunks / all labeled relevant chunks; hit rate asks whether
at least one labeled relevant chunk was retrieved. Quality averages exclude unanswerable
queries. Their retrievals remain in the JSON; this dense-only experiment does not measure
answer abstention. Compare retrieval quality with ingestion cost, token duplication, and
latency on the same dataset before selecting a configuration. No universal best chunk
size or strategy is inferred from this small local corpus.
"""
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(metadata, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    args.markdown.write_text(markdown, encoding="utf-8")
    print(table)
    print(f"\nWrote {args.output}\nWrote {args.markdown}")


if __name__ == "__main__":
    main()
