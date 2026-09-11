# Stage 24 — Chunking Benchmark

Generated from a real model run at 2026-09-10T15:40:55.062166+00:00.

```sh
/Users/aleynakilic/Documents/ChatGPT/ai/.venv/bin/python -m scripts.benchmark_chunking --device cpu
```

| Variant | Chunks | Mean tokens | Recall@5 | Hit rate@5 | MRR@5 | Ingestion ms | Index ms | Query p50/p95 ms |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| fixed_80_0 | 30 | 70.1 | 0.914 | 0.931 | 0.879 | 377.10 | 370.21 | 6.20 / 8.10 |
| fixed_80_30 | 41 | 74.8 | 0.931 | 0.931 | 0.699 | 180.31 | 172.50 | 6.00 / 6.89 |
| fixed_160_0 | 18 | 116.9 | 0.983 | 1.000 | 0.885 | 224.31 | 218.39 | 6.11 / 7.41 |
| fixed_160_30 | 18 | 131.9 | 0.983 | 1.000 | 0.756 | 164.02 | 157.58 | 6.96 / 10.84 |
| fixed_240_0 | 12 | 175.2 | 0.983 | 1.000 | 0.966 | 166.32 | 159.21 | 6.42 / 17.64 |
| fixed_240_30 | 12 | 182.7 | 0.983 | 1.000 | 0.966 | 181.98 | 173.71 | 6.03 / 9.93 |
| recursive_80_0 | 37 | 56.8 | 1.000 | 1.000 | 0.891 | 171.45 | 157.62 | 6.00 / 7.60 |
| recursive_80_30 | 37 | 62.5 | 0.983 | 1.000 | 0.856 | 179.90 | 160.62 | 6.36 / 8.04 |
| recursive_160_0 | 18 | 116.8 | 0.948 | 0.966 | 0.879 | 149.84 | 137.31 | 5.90 / 7.68 |
| recursive_160_30 | 18 | 131.8 | 0.948 | 0.966 | 0.799 | 204.41 | 189.32 | 6.44 / 7.48 |
| recursive_240_0 | 12 | 175.2 | 0.983 | 1.000 | 0.914 | 179.56 | 166.14 | 6.01 / 7.86 |
| recursive_240_30 | 12 | 182.8 | 0.983 | 1.000 | 0.914 | 179.60 | 165.42 | 5.95 / 7.07 |

## Reproducibility and controls

- Embeddings: `sentence-transformers/all-MiniLM-L6-v2` revision `1110a243fdf4706b3f48f1d95db1a4f5529b4d41`;
  dimension 384, device `cpu`, normalized vectors, batch size 32.
- Token counts: `Qwen/Qwen2.5-0.5B-Instruct` revision `c89bee90d9f811437d9735454613c35b4a3c4dc8`.
- Corpus: 9 documents, 2103 tokenizer tokens;
  SHA-256 `e825e01f5dcb22f8c78f63d821da3d15892533a3f5688f171747bf6d57335931`.
- Source labels SHA-256: `b09d7fb38c8e95e8b8784392621aa22c6cd24df2391c167a5d197941903bd400`.
- Retrieval: cosine dense search in separate in-memory Qdrant indexes, top 5;
  3 measured query passes after 1 warmup queries per index.
- JSON evidence: `/Users/aleynakilic/Documents/ChatGPT/ai/evaluation/reports/chunking_benchmark_v1.json`; includes source-resolved labels, per-query ranked IDs,
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
