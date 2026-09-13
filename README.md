# Production AI Knowledge Engine

A production-oriented LLM knowledge engine built to make retrieval quality,
grounding, context construction, inference performance, and engineering trade-offs
measurable and inspectable.

The project is being developed in tested stages. The current foundation includes
transparent attention and tokenization experiments, configurable local Hugging Face
inference, a measured dense/sparse/hybrid retrieval system, and grounded local answer
generation, an interactive frontend, and reproducible quality/performance benchmarks.
Later stages add production operations.

## Current capabilities

- Typed environment configuration
- FastAPI application factory
- Process health endpoint at `GET /health`
- OpenAPI documentation at `/docs`
- Automated formatting, linting, type-checking, and tests
- [Scaled dot-product attention from scratch](notebooks/01_attention_from_scratch.ipynb)
  with Q/K/V projections, score scaling, softmax, and causal masking experiments
- [Tokenizer inspection and comparison](notebooks/02_tokenization_inspection.ipynb)
  across English, numbers, punctuation, code, Turkish, and long words
- Configurable Hugging Face local inference using `AutoTokenizer` and
  `AutoModelForCausalLM`
- Generation controls for temperature, top-k, top-p, output length, repetition
  penalty, seed, and greedy decoding
- Deterministic Markdown/text ingestion with fixed and recursive token chunking
- Local Qdrant cosine retrieval with 384-dimensional sentence embeddings
- Okapi BM25, Reciprocal Rank Fusion, and cross-encoder reranking
- A versioned 30-query relevance set with Recall, Precision, MRR, NDCG, and latency
- Token-budgeted grounded generation with source citations and safe abstention
- Citation validation that suppresses invented sources and uncited factual claims
- Versioned FastAPI surface with stable errors, request correlation, and local-browser CORS
- Bounded four-stage system traces with timings, tokens, validation, and source lineage
- Responsive React Frontend V1 for grounded answers, evidence cards, and trace inspection
- Browser document uploads with UTF-8 Markdown/text validation, persistent storage,
  duplicate detection, and automatic retrieval refresh
- Backwards-compatible local RAG API at `POST /rag/answer`
- Pinned embedding model and batch-size comparisons with retrieval quality and encoding cost
- Fixed/recursive chunking parameter sweeps with freshly resolved source labels
- Repeated RAG evaluation with citation, abstention, lexical answer checks, and phase timings
- Generation latency distributions across prompt lengths and output caps

## Prerequisites

- [`uv`](https://docs.astral.sh/uv/)
- Git

`uv` installs and manages the pinned Python 3.12 runtime, so a separate Python
installation is not required.

## Local setup

```bash
git clone <repository-url>
cd production-ai-knowledge-engine
uv sync --all-groups
cp .env.example .env
```

Run the API:

```bash
uv run uvicorn backend.app.main:app --reload
```

Then inspect:

- Health: <http://127.0.0.1:8000/health>
- Versioned health: <http://127.0.0.1:8000/api/v1/health>
- OpenAPI UI: <http://127.0.0.1:8000/docs>

Run Frontend V1 in a second terminal:

```bash
cd frontend
npm install
npm run dev
```

Open <http://localhost:3000>. The local API and frontend do not require an external AI API key.

### Workspaces and documents

Create an account with your email, password, and workspace name in the console. Owners can
add existing accounts as editors or readers. Documents, retrieval, answers, caches, sources,
and traces are isolated by workspace and authorized on every request.

Upload PDF, DOCX, Markdown, or UTF-8 text files. Uploads return immediately after durable
admission and progress through **Processing → Ready / Failed** in the document panel.
You can keep asking questions about ready documents while background processing runs.
PDF citations open the relevant page's extracted text; DOCX citations open a section.
Original source files can be downloaded. Image-only PDFs require OCR before uploading.

Use **Replace** to upload a new version, **Retry** after a processing failure, or **Delete**
to remove a document and its versions. Failed replacements keep the previous ready version.
Indexing embeds only added/replaced documents and updates their vectors; successful changes
invalidate the affected workspace's answer cache.

Existing shared uploads are preserved and require an explicit administrator import into a
chosen workspace. They are never automatically exposed to new accounts. The bundled NovaStack
corpus remains available to benchmark/CLI workflows, not private workspace retrieval.

See [Workspace setup, API, limits, and migration](docs/workspaces.md) for account provisioning,
permissions, cookie deployment requirements, background-job recovery, and legacy imports.
Use `localhost` for both local services; the default API URL is `http://localhost:8000`.

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
```

The frontend is also checked in GitHub Actions. Run its checks locally with:

```bash
cd frontend
npm run lint
npm run typecheck
npm run build
```

To apply formatting locally:

```bash
uv run ruff format .
```

Execute the Stage 1 notebook and save fresh outputs:

```bash
uv run jupyter execute --inplace --timeout=120 \
  notebooks/01_attention_from_scratch.ipynb
```

Execute the Stage 2 tokenizer comparison:

```bash
uv run jupyter execute --inplace --timeout=180 \
  notebooks/02_tokenization_inspection.ipynb
```

The pytest suite also executes the notebook from a clean in-memory copy, so
notebook assertions and hidden state cannot silently break continuous integration.

## Local model generation

The first run downloads the pinned model weights to the Hugging Face cache:

```bash
uv run python -m scripts.generate \
  --prompt "Explain vector search in one paragraph." \
  --temperature 0.7 \
  --top-p 0.9 \
  --max-new-tokens 96
```

Run the reproducible Stage 4 sampling comparison:

```bash
uv run python -m scripts.compare_generation
```

See [the model decision](docs/model-choice.md), the measured
[local inference evidence](docs/local-inference.md), and the generated
[sampling report](docs/generation-sampling.md) for the measured trade-offs.

## Retrieval pipeline

Create both chunking variants and their resolved relevance labels:

```bash
uv run python -m scripts.ingest data/raw
```

This command prepares the benchmark corpus and resolves its bundled relevance labels.
It validates every selected chunking strategy before replacing existing chunk and label
files. Use the document upload API for additional user documents. The search CLI rebuilds
its dense collection from the current prepared chunks on each invocation to avoid stale
results after re-ingestion or an embedding-model change.

Build local Qdrant indexes and run the complete retrieval benchmark:

```bash
uv run python -m scripts.evaluate_retrieval
```

Search interactively with `dense`, `bm25`, `hybrid`, or `reranked`:

```bash
uv run python -m scripts.search \
  "What caused the authentication incident?" \
  --method reranked
```

See the [retrieval architecture](docs/retrieval-system.md), the generated
[chunking comparison](docs/chunking-comparison.md), and the measured
[retrieval report](docs/retrieval-evaluation.md).

## Grounded RAG

Ask a source-grounded question with the pinned local language model:

```bash
uv run python -m scripts.rag "How long do access tokens last?"
```

Run the 20-question answer, citation, grounding, and abstention benchmark:

```bash
uv run python -m scripts.evaluate_rag
```

The workspace pipeline is available at authenticated `POST /rag/answer`; the CLI uses the
benchmark corpus. By default, API startup loads the
embedding model, reranker, and language model and builds the retrieval index before
accepting requests. The first launch may download pinned model weights; later launches
reuse the download cache. Startup fails if preparation fails. This moves loading cost
to startup; it does not eliminate inference latency or execute a warmup generation.
Set `PAKE_RAG_PRELOAD_ON_STARTUP=false` to opt into lazy loading for lightweight development.

Repeated questions reuse validated answers through a process-local LRU cache (256 entries,
15-minute TTL by default). Keys distinguish the document revision, question (outer whitespace
trimmed, otherwise exact), and effective `top_k`. Each workspace has its own cache and
membership is checked before access. The cache belongs to one runtime/configuration;
changing model, prompt, or retrieval settings requires restarting the API.
Successfully indexed uploads invalidate the affected workspace cache; duplicate uploads retain them.
Rejected, abstained, and invalid answers are not cached. Set `PAKE_RAG_CACHE_MAX_ENTRIES=0`
to disable caching. Cached responses retain citations, expose `cache_hit=true`, report no new
generation tokens, and get fresh request/trace IDs. Restarting clears the cache.

See the [Grounded RAG design](docs/grounded-rag.md) and current
[RAG evaluation](docs/rag-evaluation.md). The earlier
[Stage 13–16 report](docs/grounded-rag-evaluation.md) is retained as historical evidence.

## Evaluation and performance — Stages 23–26

Run the four benchmarks from the repository root. Each command writes real local model
measurements to JSON and a readable Markdown report. They use isolated evaluation indexes;
the serving Qdrant collection is not changed. First runs download any missing pinned models.

```bash
# Stage 23: MiniLM L6/L12, batch sizes 8/32, identical corpus and labels
uv run python -m scripts.benchmark_embeddings --device cpu

# Stage 24: fixed/recursive × 80/160/240 tokens × 0/30 overlap
uv run python -m scripts.benchmark_chunking --device cpu

# Stage 25: end-to-end answers, citations, abstention, and stage timings
uv run python -m scripts.evaluate_rag --device cpu --retrieval-device cpu

# Stage 26: short/medium/long prompts × 32/64 output-token caps
uv run python -m scripts.benchmark_generation --device cpu
```

Use `--help` for custom repeats, warmups, input datasets, and output destinations. Set
`HF_HUB_OFFLINE=1` to use cached models without network access. Run benchmarks sequentially
on an otherwise idle machine; overlapping inference jobs distort latency comparisons.

| Stage | Report | Machine-readable evidence |
|---|---|---|
| 23 | [Embedding Benchmark](docs/embedding-benchmark.md) | [JSON](evaluation/reports/embedding_benchmark.json) |
| 24 | [Chunking Benchmark](docs/chunking-benchmark.md) | [JSON](evaluation/reports/chunking_benchmark_v1.json) |
| 25 | [RAG Evaluation](docs/rag-evaluation.md) | [JSON](evaluation/reports/rag_evaluation_v1.json) |
| 26 | [Generation Latency Benchmark](docs/generation-latency-benchmark.md), [measured results](evaluation/reports/generation_latency.md) | [JSON](evaluation/reports/generation_latency.json) |

Stage 23–24 report conventional Recall (fraction of relevant labeled chunks found) and
Hit Rate (any relevant hit) separately. The old schema-v1 retrieval artifact called Hit
Rate “Recall”; it is kept as historical evidence. RAG correctness is an accepted-answer
lexical proxy, and the grounding gate is not an independent semantic judge. Generation
throughput includes prefill and decoding; TTFT is explicitly unavailable. Model/data
fingerprints, timing boundaries, raw trials, and limitations are included in the reports.

## Configuration

Runtime settings are read from environment variables prefixed with `PAKE_`.
Copy `.env.example` to `.env` for local overrides. The `.env` file is ignored by
Git; only placeholder values belong in `.env.example`.

| Variable | Default | Purpose |
|---|---|---|
| `PAKE_APP_NAME` | `Production AI Knowledge Engine` | API display name |
| `PAKE_ENVIRONMENT` | `development` | Runtime environment |
| `PAKE_LOG_LEVEL` | `INFO` | Application log severity |
| `PAKE_MODEL_NAME` | `Qwen/Qwen2.5-0.5B-Instruct` | Hugging Face model repository |
| `PAKE_MODEL_REVISION` | pinned commit | Immutable model revision |
| `PAKE_DEVICE` | `auto` | `auto`, `cpu`, `mps`, or `cuda` |
| `PAKE_MAX_NEW_TOKENS` | `128` | Default generation limit |
| `PAKE_CHUNK_STRATEGY` | `recursive` | Uploaded-document chunking strategy: `fixed` or `recursive` |
| `PAKE_CHUNK_SIZE_TOKENS` | `160` | Maximum chunk token count |
| `PAKE_CHUNK_OVERLAP_TOKENS` | `30` | Token overlap between chunks |
| `PAKE_EMBEDDING_MODEL_NAME` | `sentence-transformers/all-MiniLM-L6-v2` | Dense embedding model |
| `PAKE_RERANKER_MODEL_NAME` | `cross-encoder/ms-marco-MiniLM-L6-v2` | Candidate reranker |
| `PAKE_RETRIEVAL_DEVICE` | `auto` | Retrieval model accelerator |
| `PAKE_RETRIEVAL_TOP_K` | `5` | Final retrieval depth |
| `PAKE_RETRIEVAL_CANDIDATE_K` | `20` | Hybrid/reranker candidate depth |
| `PAKE_RAG_TOP_K` | `5` | Retrieved passages sent to context assembly |
| `PAKE_RAG_MAX_SOURCES` | `3` | Maximum source blocks in one answer |
| `PAKE_RAG_CONTEXT_TOKENS` | `650` | Source context token budget |
| `PAKE_RAG_MAX_NEW_TOKENS` | `80` | Grounded answer generation limit |
| `PAKE_RAG_STRICT_GROUNDING` | `true` | Suppress answers that fail citation validation |
| `PAKE_RAG_MIN_RETRIEVAL_SCORE` | `0.8` | Cross-encoder confidence floor for generation |
| `PAKE_RAG_EXTRACTIVE_FALLBACK_SCORE` | `1.0` | Minimum score for cited extractive fallback |
| `PAKE_RAG_PRELOAD_ON_STARTUP` | `true` | Prepare models and index before accepting requests |
| `PAKE_RAG_CACHE_MAX_ENTRIES` | `256` | In-process LRU answer capacity; `0` disables caching |
| `PAKE_RAG_CACHE_TTL_SECONDS` | `900` | Answer lifetime from insertion; hits do not extend it |
| `PAKE_API_CORS_ORIGINS` | local ports 3000 and 5173 | Comma-separated allowed browser origins |
| `PAKE_TRACE_MAX_RECORDS` | `200` | Maximum recent in-process system traces |
| `PAKE_DOCUMENTS_PATH` | `data/uploads/documents.sqlite3` | Preserved legacy shared library; used by import/CLI |
| `PAKE_WORKSPACES_PATH` | `data/workspaces` | Accounts, sessions, isolated libraries, queues, and indexes |
| `PAKE_AUTH_SESSION_SECONDS` | `28800` | Session lifetime |
| `PAKE_AUTH_ALLOW_REGISTRATION` | `true` | Allow account creation from the console |
| `PAKE_UPLOAD_MAX_EXPANDED_BYTES` | `20971520` | Expanded/extracted document limit |
| `PAKE_UPLOAD_MAX_PAGES` | `500` | Maximum PDF pages |
| `PAKE_UPLOAD_MAX_VERSIONS` | `10` | Maximum versions per document |
| `PAKE_UPLOAD_MAX_BYTES` | `5242880` | Maximum bytes per uploaded file |
| `PAKE_UPLOAD_MAX_DOCUMENTS` | `100` | Maximum documents per workspace |
| `PAKE_UPLOAD_MAX_CHUNKS` | `2000` | Maximum active chunks per workspace |

## Repository structure

```text
backend/
  app/
    config.py       # Typed environment configuration
    documents/      # Legacy document library and shared validation
    workspaces/     # Accounts, authorization, source parsing, durable queue, and isolated RAG
    observability/  # Structured system traces and bounded in-process storage
    embeddings/     # Sentence embedding boundary and implementation
    evaluation/     # Query labels, quality metrics, and performance benchmarks
    ingestion/      # Loaders, cleaning, token chunking, and persistence
    main.py         # FastAPI application and health endpoint
    llm/            # Model loading, prompt construction, and generation
    rag/            # Context budgeting, grounded prompts, citations, and answer service
    retrieval/      # Qdrant, BM25, RRF, hybrid search, and reranking
data/raw/           # Versioned NovaStack demonstration knowledge base
data/uploads/       # Preserved legacy shared library (ignored by Git)
data/workspaces/    # Private accounts, documents, jobs, and indexes (ignored by Git)
data/evaluation/    # Human-reviewable and resolved retrieval relevance labels
scripts/            # Reproducible CLI and experiment runners
docs/               # Decisions and measured engineering reports
evaluation/reports/ # Machine-readable experiment evidence
tests/              # Backend tests
notebooks/          # Executable AI/LLM learning experiments
frontend/           # React knowledge console and system trace UI
.github/workflows/  # Continuous integration
```

Generated chunk files and the local Qdrant database are ignored; rerunning ingestion and
evaluation recreates them from versioned sources.
TypeScript build caches and the unrelated local `github-profile-draft/` directory are
also ignored. Keep notebooks, labeled datasets, lockfiles, and historical benchmark
reports: they document the staged experiments and make comparisons reproducible.
