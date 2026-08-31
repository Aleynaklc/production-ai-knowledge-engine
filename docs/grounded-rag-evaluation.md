# Grounded RAG Evaluation

This report is generated from 20 versioned questions: 16 answerable and 4 deliberately
unanswerable. Required answer terms provide a small deterministic correctness check;
source labels measure citation precision and recall. All generation uses the pinned local
model with greedy decoding.

Citation precision is averaged over answerable responses that contain citations. Citation
recall includes every answerable query, so abstentions reduce recall but not precision.

| Metric | Value |
|---|---:|
| Answer correctness | 0.688 |
| Citation precision | 0.917 |
| Citation recall | 0.688 |
| Grounded answer rate | 0.750 |
| Answerable response rate | 0.750 |
| Abstention accuracy | 1.000 |
| Safe unanswerable rate | 1.000 |
| Rejection rate | 0.000 |
| Extractive fallback rate | 0.600 |
| Mean end-to-end latency | 1912.22 ms |
| p95 end-to-end latency | 3563.85 ms |

## Safety behavior

Answers without citations, with invented source identifiers, or with uncited/unsupported
factual claims are not returned to the caller. Strict mode first attempts a high-confidence
verbatim source fallback and otherwise returns the standard insufficient-context response.
It retains `raw_answer` and validation issues for debugging.

## Diagnostics

- Incorrect or suppressed answerable queries: `['r003', 'r006', 'r008', 'r010', 'r011']`
- Unsafe answers to unanswerable queries: `[]`
- Outputs rejected by the grounding gate: `[]`

Detailed answers, citations, timings, and failure reasons are stored in
`evaluation/reports/grounded_rag_v1.json`.
