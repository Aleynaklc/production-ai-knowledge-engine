# Grounded RAG (Stages 13–16)

> The production API now uses email/password accounts, workspace authorization, asynchronous
> document processing, PDF/DOCX sources, and versioned document management. See the
> [current workspace contract](workspaces.md). Shared API examples below describe the earlier
> implementation/benchmark harness and are not an unauthenticated production endpoint.

The grounded RAG layer converts ranked retrieval results into answers that can be audited
and safely rejected. It runs entirely on local models by default and does not require an
external AI API key.

## Execution path

1. Hybrid retrieval and cross-encoder reranking produce the candidate passages.
   Queries below the configured reranker confidence floor abstain before generation.
2. The context builder deduplicates passages, assigns stable `[S1]`, `[S2]` identifiers,
   and enforces the configured tokenizer budget.
3. Source text is escaped and wrapped in explicit source blocks. The system prompt treats
   it as untrusted data to reduce the risk of document instructions overriding RAG rules.
   This prompting boundary is not a guarantee against prompt injection.
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

Rejected API responses retain `raw_answer` and validation issues for diagnostics; the
user-facing `answer` field contains the insufficient-context message. The lexical gate
recognizes Unicode words but is not an independent semantic correctness judge.

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

API startup loads the embedding model, reranker, and language model and prepares the
current index before accepting requests. Preparation runs in a worker thread; failures
abort startup and close owned storage resources. No answer is generated during preparation.
Set `PAKE_RAG_PRELOAD_ON_STARTUP=false` to load on the first question instead.
Requests reuse these model instances, and blocking work runs in a worker thread.

The service caches only validated `answered` results in a bounded in-memory LRU cache.
The default capacity is 256 and TTL is 900 seconds from insertion, configurable through
`PAKE_RAG_CACHE_MAX_ENTRIES` (zero disables) and `PAKE_RAG_CACHE_TTL_SECONDS`.
Questions match exactly after trimming outer whitespace, with the same effective `top_k`
and uploaded document revision. A configuration snapshot and base corpus belong to one
service lifetime: restart after changing model/prompt/settings or bundled files. There is
no semantic matching and no cache shared across processes. Upload revision checks happen
before every cache lookup; a failed refresh raises an error instead of serving stale data.
Duplicate uploads do not invalidate the cache. Stored answers are deep-copied so callers
cannot alter future responses. Concurrent identical requests share the existing runtime
lock and generate only once when a valid answer can be cached.

## Evaluation

```bash
uv run python -m scripts.evaluate_rag
```

The benchmark covers answer content, citation precision/recall, grounding validity,
abstention on unanswerable questions, rejection behavior, and end-to-end latency. Results
are now written to `evaluation/reports/rag_evaluation_v1.json` and
`docs/rag-evaluation.md`. The earlier `grounded_rag_v1.json` and
`grounded-rag-evaluation.md` files are retained as historical Stage 13–16 evidence.
