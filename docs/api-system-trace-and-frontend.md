# API, system trace, and Frontend V1

Stages 18–20 expose the grounded RAG pipeline through a versioned HTTP API, record a
bounded execution trace for every answer, and render the result in a responsive web console.
All inference remains local; no external model API key is required.

## FastAPI surface

The application keeps the original `GET /health` and `POST /rag/answer` routes for backwards
compatibility. New clients should use the versioned surface:

| Method | Path | Purpose |
|---|---|---|
| `GET` | `/api/v1/health` | Process health and API version without loading models |
| `GET` | `/api/v1/system` | Safe runtime capabilities for the frontend |
| `GET` | `/api/v1/documents` | Uploaded documents, supported extensions, and file-size limit |
| `POST` | `/api/v1/documents?filename=guide.md` | Validate, chunk, and persist raw file bytes |
| `POST` | `/api/v1/answers` | Grounded answer plus its complete system trace |
| `GET` | `/api/v1/traces` | Newest-first recent traces; `limit` is 1–100 |
| `GET` | `/api/v1/traces/{trace_id}` | One immutable trace by identifier |

Every response includes `X-Request-ID`. Successful answer responses include the same
`request_id`, a unique `trace_id`, the grounded `result`, and the trace snapshot. Validation,
bad-request, and trace-not-found failures use a stable `{ "error": { ... } }` envelope and do
not expose internal exception details.
Whitespace-only questions fail validation before loading models. Model, inference, and
index failures return HTTP 503 with `rag_unavailable`; document-storage failures also
return HTTP 503. Failed requests do not create successful execution traces.

Browser origins are configured with the comma-separated `PAKE_API_CORS_ORIGINS` setting.
Defaults allow the local frontend development ports 3000 and 5173 only.
Health responses confirm that the API process is running; they do not establish that
model loading or the first answer will succeed. Deployment checks should include a real
upload and answer request.

## Document upload contract

Send a UTF-8 `.md` or `.txt` file as the raw request body, with its filename in the
`filename` query parameter. The browser uses `Content-Type: application/octet-stream`;
`text/plain` and `text/markdown` are also accepted. Multipart form bodies are unsupported.
The server checks the declared length and enforces the byte limit while reading the stream,
including requests without a `Content-Length` header.

A new upload returns HTTP 201 with `{ "document": { ... }, "duplicate": false }`.
Uploading identical bytes returns HTTP 200 with the original document and `duplicate: true`.
The document contains its ID, display filename, title, byte size, SHA-256, creation time,
chunk count, chunking settings, and `status: "ready"`. Ready means validation, chunking,
and persistence have completed; the next answer refreshes retrieval before searching.

`GET /api/v1/documents` returns `documents`, `count`, `max_file_bytes`,
`supported_extensions`, and `scope: "shared"`. Listing documents does not load any models.
The API has a shared library; it does not provide authentication or per-user isolation.
The frontend states this next to the upload form.

| Status | Upload failure |
|---|---|
| 400 | Invalid filename or content-length header |
| 409 | Document count or total uploaded-chunk capacity exceeded |
| 413 | File exceeds the configured byte limit |
| 415 | Unsupported extension or content type |
| 422 | Missing filename, invalid UTF-8, binary content, or empty document |
| 503 | Document processing or storage unavailable |

Errors use the same request-correlated envelope as the answer API. Defaults are 5 MiB
per file, 100 documents, and 2,000 total uploaded chunks. Set `PAKE_UPLOAD_MAX_BYTES`,
`PAKE_UPLOAD_MAX_DOCUMENTS`, and `PAKE_UPLOAD_MAX_CHUNKS` to adjust these limits.

Documents and chunks are committed together in SQLite at `PAKE_DOCUMENTS_PATH`
(`data/uploads/documents.sqlite3` by default). Concurrent uploads are deduplicated and
quota-checked in the write transaction. Failed processing does not leave a partial document.
The stored chunking settings describe the upload: `PAKE_CHUNK_STRATEGY`,
`PAKE_CHUNK_SIZE_TOKENS`, and `PAKE_CHUNK_OVERLAP_TOKENS` control future uploads.

On the next question, the RAG service combines uploaded chunks with the bundled corpus,
builds replacement dense and BM25 indexes, and swaps them into the service only after indexing
succeeds. It reuses the loaded embedding, reranking, and language models. Answer execution
and index refresh are serialized within the process. Serving collections are separate from
benchmark collections. Run one API worker when using the local Qdrant storage directory.

## System trace

Each answer trace records four ordered stages:

1. retrieval;
2. context assembly;
3. generation;
4. grounding and citation validation.

The record includes measured stage and total latency, source lineage for returned citations,
context/input/output token counts, validation issues, fallback use, and the final safety status.
Traces are stored in a thread-safe in-process store. The newest 200 records are retained by
default (`PAKE_TRACE_MAX_RECORDS`); the oldest record is evicted when the bound is reached.
The store contains the submitted question, source metadata, and diagnostics; it does not
contain the full assembled model prompt, model weights, or hidden chain-of-thought.
Trace routes share the API's unauthenticated access, so restrict access to the service
before using confidential questions or documents. Restarting the API clears the store.

## Frontend V1

The frontend is in `frontend/` and uses React, TypeScript, vinext, Tailwind CSS, shadcn
components, and Lucide icons. It supports answer, abstention, rejection, loading, API-offline,
and empty states. The main workspace renders inline citation markers, evidence cards, measured
pipeline timings, token counts, validation status, and trace correlation metadata.
The document panel supports file selection, upload progress, errors, duplicate feedback,
and a persistent library listing. Successful uploads clear the previous answer so the next
question uses the updated library.

Start both processes in separate terminals:

```bash
uv run uvicorn backend.app.main:app --reload
```

```bash
cd frontend
npm install
npm run dev
```

The frontend uses `http://127.0.0.1:8000` by default. Copy `frontend/.env.example` to
`frontend/.env.local` to point it at another API with `NEXT_PUBLIC_API_BASE_URL`.

## Verification

Backend contracts and trace retention are covered by `tests/test_api_tracing.py`.
Upload contract and persistence/refresh checks are in `tests/test_document_upload_api.py`
and `tests/documents/`. Run the complete backend and frontend checks with:

```bash
uv run ruff format --check .
uv run ruff check .
uv run mypy
uv run pytest
cd frontend
npm run lint
npm run typecheck
npm run build
```

The upload feature was also checked with the pinned local models on CPU, using isolated
temporary document and Qdrant stores. Uploading `borealis.md` returned the correct
`BLUE-7429` escalation code with that document's citation. Uploading `altair.md` after
the first answer returned its `03:40 UTC` maintenance start time with the new citation,
while retaining the same loaded generator. A new application instance listed both saved
documents. This verifies the upload-to-answer integration and restart persistence; it is
not a general answer-quality benchmark.
