# Production AI Knowledge Engine

A production-oriented LLM knowledge engine built to make retrieval quality,
grounding, context construction, inference performance, and engineering trade-offs
measurable and inspectable.

The project is being developed in tested stages. The current foundation includes
transparent attention and tokenization experiments, configurable local Hugging Face
inference, a measured dense/sparse/hybrid retrieval system, and grounded local answer
generation. Later stages add the interactive frontend and production operations.

## Current capabilities

- Typed environment configuration
- FastAPI application factory
- Readiness endpoint at `GET /health`
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
- Lazy local RAG API at `POST /rag/answer`

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
- OpenAPI UI: <http://127.0.0.1:8000/docs>

## Quality checks

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
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

The same pipeline is available at `POST /rag/answer`. Models are loaded lazily on the
first RAG request. See the [Grounded RAG design](docs/grounded-rag.md) and generated
[Grounded RAG evaluation](docs/grounded-rag-evaluation.md).

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

## Repository structure

```text
backend/
  app/
    config.py       # Typed environment configuration
    embeddings/     # Sentence embedding boundary and implementation
    evaluation/     # Versioned query labels and ranking metrics
    ingestion/      # Loaders, cleaning, token chunking, and persistence
    main.py         # FastAPI application and health endpoint
    llm/            # Model loading, prompt construction, and generation
    rag/            # Context budgeting, grounded prompts, citations, and answer service
    retrieval/      # Qdrant, BM25, RRF, hybrid search, and reranking
data/raw/           # Versioned NovaStack demonstration knowledge base
data/evaluation/    # Human-reviewable and resolved retrieval relevance labels
scripts/            # Reproducible CLI and experiment runners
docs/               # Decisions and measured engineering reports
evaluation/reports/ # Machine-readable experiment evidence
tests/              # Backend tests
notebooks/          # Executable AI/LLM learning experiments
.github/workflows/  # Continuous integration
```

Generated chunk files and the local Qdrant database are ignored; rerunning ingestion and
evaluation recreates them from versioned sources.
