# Production AI Knowledge Engine

A production-oriented LLM knowledge engine built to make retrieval quality,
grounding, context construction, inference performance, and engineering trade-offs
measurable and inspectable.

The project is being developed in tested stages. The current foundation includes
transparent attention and tokenization experiments plus configurable local Hugging
Face inference and sampling benchmarks. Later stages add document ingestion, dense
and sparse retrieval, reranking, grounded generation, evaluation, and an interactive
frontend.

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

## Repository structure

```text
backend/
  app/
    config.py       # Typed environment configuration
    main.py         # FastAPI application and health endpoint
    llm/            # Model loading, prompt construction, and generation
scripts/            # Reproducible CLI and experiment runners
docs/               # Decisions and measured engineering reports
evaluation/reports/ # Machine-readable experiment evidence
tests/              # Backend tests
notebooks/          # Executable AI/LLM learning experiments
.github/workflows/  # Continuous integration
```

Additional directories will be introduced only when their implementation stage
begins, keeping the repository runnable and avoiding empty architecture.
