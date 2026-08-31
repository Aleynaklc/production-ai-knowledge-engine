# Retrieval System (Stages 5–11)

The retrieval subsystem turns local Markdown and text documents into measurable search
results. Its boundaries are deliberately small: loaders create typed documents, chunkers
create typed passages, embedding providers return plain vectors, and every retriever
returns the same ranked result contract.

## Pipeline

1. Normalize UTF-8 source files and preserve source/title metadata.
2. Produce fixed-token and structure-aware recursive chunks with deterministic UUIDs.
3. Embed chunks with `sentence-transformers/all-MiniLM-L6-v2` (384 dimensions).
4. Store vectors and source payloads in a persistent local Qdrant collection.
5. Build an in-memory Okapi BM25 index over the same chunks.
6. Merge dense and lexical candidates using Reciprocal Rank Fusion:
   `RRF(d) = sum(1 / (60 + rank_i(d)))`.
7. Score hybrid candidates with `cross-encoder/ms-marco-MiniLM-L6-v2`.
8. Evaluate Recall, Precision, MRR, NDCG, and latency against versioned labels.

The default embedding model is compact enough for local development and emits normalized
384-dimensional vectors. Cosine similarity is therefore explicit in the Qdrant schema.
The cross-encoder is only applied to the small candidate set because it jointly processes
the query and passage and is more expensive than first-stage retrieval.

## Reproduce

From the repository root:

```bash
uv sync --dev
uv run python -m scripts.ingest data/raw
uv run python -m scripts.evaluate_retrieval
uv run python -m scripts.search "How long do access tokens last?" --method reranked
```

The first command writes both chunking strategies and resolves their labels. The second
recreates the local indexes, evaluates five configurations, and writes:

- `evaluation/reports/ingestion_v1.json`
- `evaluation/reports/retrieval_v1.json`
- `docs/chunking-comparison.md`
- `docs/retrieval-evaluation.md`

`data/qdrant/` and `data/processed/` are generated and ignored. The raw corpus, source
queries, resolved relevance labels, and reports are versioned so the evidence remains
reviewable.
