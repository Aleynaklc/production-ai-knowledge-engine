# Automatic generation provider selection

The workspace API uses `PAKE_GENERATION_PROVIDER=auto` by default:

- A nonempty `OPENAI_API_KEY` selects OpenAI for generated answers.
- An absent, empty, or whitespace-only key keeps the existing local Hugging Face model.
- `PAKE_GENERATION_PROVIDER=local` forces local generation even if a key is present.
- `PAKE_GENERATION_PROVIDER=openai` requires a key and fails configuration validation otherwise.

Add these settings to the **backend** `.env` file and restart the backend:

```dotenv
PAKE_GENERATION_PROVIDER=auto
OPENAI_API_KEY=your-api-key
PAKE_OPENAI_MODEL=gpt-4.1-mini
PAKE_OPENAI_TIMEOUT_SECONDS=30
```

Leave `OPENAI_API_KEY=` blank to use the local model. `PAKE_OPENAI_API_KEY` is also
accepted; use only one key variable. When both are supplied by the same configuration
source, `OPENAI_API_KEY` takes precedence. Never put a provider key into frontend
environment variables or commit it. The checked-in `.env.example` contains no credentials.

Selection checks whether a key is configured, not whether the provider will accept it.
Authentication and model access are checked by the provider on the first uncached
generation request. There is no startup API call or automatic retry. An invalid key,
quota limit, network failure, timeout, refusal, or incomplete response produces a
sanitized error instead of silently falling back to the local model. Remove the key
or select `local`, then restart, to return to local generation.

## What changes

OpenAI receives the grounded generation prompt, including the user's question and
selected document passages. Parsing, chunking, embeddings, vector storage, hybrid
retrieval, reranking, access checks, and citation validation remain local. The adapter
uses the [Responses API](https://developers.openai.com/api/reference/cli/resources/responses/methods/create)
with `store: false`; this flag is not a promise of zero retention. The configurable
default is [GPT-4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini).

API mode does not load the local generation model's weights. It still needs the local
retrieval models and the existing Hugging Face tokenizer. Keeping that tokenizer
preserves chunk boundaries and context budgeting, so switching providers does not
require reindexing. Context token counts are local estimates; generated-answer input
and output token counts come from the provider's response. A provider context-limit
error is reported explicitly, without silently truncating the evidence.

Answers still pass the existing grounding checks. An API model does not guarantee a
correct answer, repair missing source evidence, or bypass a rejected answer. Cache
hits and supported extractive/calculation paths may answer without calling either
generation model. Answer caches are scoped to the workspace, source revision, document
selection, provider, and model. Errors are not cached as answers.

Authenticated `GET /api/v1/system` reports `inference`, `model`, `provider_selection`,
and `server_api_key_configured`, never the key. Generated answer records and generation
traces identify the provider and model. Existing local CLI benchmarks, including
`scripts.evaluate_workspaces`, stay local even when an API key is configured.

## Verification

`tests/rag/test_remote.py` uses mocked HTTP transport to cover selection, key redaction,
the Responses request contract, usage, failures, and local-weight loading. The workspace
integration test covers safe HTTP errors, retrying after failure, source-bearing answers,
and cache reuse. These tests make no paid API calls and do not establish live model quality.
