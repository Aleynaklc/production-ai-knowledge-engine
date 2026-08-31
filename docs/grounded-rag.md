# Grounded RAG (Stages 13–16)

The grounded RAG layer converts ranked retrieval results into answers that can be audited
and safely rejected. It runs entirely on local models by default and does not require an
external AI API key.

## Execution path

1. Hybrid retrieval and cross-encoder reranking produce the candidate passages.
   Queries below the configured reranker confidence floor abstain before generation.
2. The context builder deduplicates passages, assigns stable `[S1]`, `[S2]` identifiers,
   and enforces the configured tokenizer budget.
3. Source text is escaped and wrapped in explicit source blocks. The system prompt treats
   it as untrusted data, preventing instructions in documents from overriding RAG rules.
4. The pinned local Qwen model generates with greedy decoding and must cite every factual
   sentence.
5. The grounding gate parses citations, checks that every identifier exists, and rejects
   uncited or lexically unsupported factual claims.
6. Missing evidence produces a fixed insufficient-context response rather than a guess.
7. If generation fails validation, a high-confidence cross-encoder may select one verbatim
   source sentence as a cited extractive fallback; otherwise the answer remains rejected.

Strict validation has three public states:

- `answered`: the answer contains only recognized citations and every claim is cited.
- `abstained`: no answer was claimed because evidence was insufficient.
- `rejected`: the model produced an answer, but the grounding gate suppressed it.

Rejected responses retain `raw_answer` and validation issues for offline diagnostics; the
caller receives the safe insufficient-context message.

## Run locally

```bash
uv run python -m scripts.rag "How long do access tokens last?"
```

Start the HTTP API and call the same pipeline:

```bash
uv run uvicorn backend.app.main:app --reload
curl -X POST http://127.0.0.1:8000/rag/answer \
  -H 'Content-Type: application/json' \
  -d '{"question":"What caused incident 2026-001?"}'
```

The first request lazily loads the embedding model, reranker, and language model. Later
requests reuse those model instances. The endpoint performs blocking model work in a
worker thread so it does not block the FastAPI event loop.

## Evaluation

```bash
uv run python -m scripts.evaluate_rag
```

The benchmark covers answer content, citation precision/recall, grounding validity,
abstention on unanswerable questions, rejection behavior, and end-to-end latency. Results
are written to `evaluation/reports/grounded_rag_v1.json` and
`docs/grounded-rag-evaluation.md`.
