# Chunking Comparison

Historical artifact: `Recall@5` below is the earlier Hit Rate definition. See
[Stage 24](chunking-benchmark.md) for the expanded parameter grid with conventional
Recall and separate Hit Rate measurements.

Both strategies use 160 tokenizer tokens with
30 tokens of overlap. Fixed chunking slices token windows;
recursive chunking first preserves Markdown blocks and sentence boundaries, then falls
back to token windows for oversized units.

| Dense configuration | Recall@5 | MRR@5 | Mean latency (ms) |
|---|---:|---:|---:|
| Fixed token | 1.000 | 0.756 | 13.79 |
| Recursive | 0.966 | 0.799 | 10.95 |

The corpus, chunking code, relevance labels, and reports are committed so this comparison
can be reproduced after any tokenizer or chunk-size change.
