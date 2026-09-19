# Authenticated workspaces and document processing

The production entry point (`backend.app.main:app`) now requires email/password sessions
and workspace membership for documents, sources, answers, system details, and traces.
The historical shared API harness exists only under `tests/legacy_api.py`; it is not mounted.
The benchmark CLI and curated NovaStack datasets remain independent of user workspaces.

## First use

1. Start the backend with one worker: `uv run uvicorn backend.app.main:app --reload`.
2. Start the frontend and use `localhost` consistently for both services. The default
   frontend API URL is `http://localhost:8000`. Do not mix `localhost` and `127.0.0.1`
   for session-cookie requests.
3. Choose **Create an account**, enter an email, a password of at least 12 characters,
   and a workspace name. This creates an owner membership in a new, empty workspace.
4. Upload a PDF, DOCX, Markdown, or text file. Wait for **Ready**, then ask a question.
5. An owner can add an existing account as a reader or editor under **Manage workspace
   members**, change that role, and revoke access. The other account refreshes its session
   (sign out/in) to see newly added workspaces. Authorization is rechecked on every request.

Self-registration can be disabled with `PAKE_AUTH_ALLOW_REGISTRATION=false`. Local operators
can provision an account without emailing or printing its password:

```bash
uv run python -m scripts.create_account --email owner@example.com --workspace 'Company A'
```

This release implements local password accounts, not email verification, password reset
email, SSO, or MFA. Email is a login identifier and is not claimed to be verified. Protect
production account creation through configuration and your deployment's access controls.

## Access boundary

Passwords use independent random salts and scrypt hashes. Session cookies are HttpOnly,
SameSite=Lax, and expire after eight hours by default. Only session token hashes are stored.
Cookies are Secure when `PAKE_ENVIRONMENT=production`; production therefore requires HTTPS.
Use a same-site frontend/API deployment (for example `app.example.com` and `api.example.com`)
or a same-origin reverse proxy. Arbitrary cross-site hosting does not work with Lax cookies.
Configure allowed browser origins through `PAKE_API_CORS_ORIGINS`; cross-origin mutation
requests from other origins are rejected. Login and registration share an in-process,
bounded per-client-address limit of 10 attempts per minute. Forwarded IP headers are not trusted.

The frontend sends `X-Workspace-ID`, but the server checks the authenticated user's membership;
that header alone grants no access. Owners/editors may upload, replace, retry, and delete.
Readers may list documents, ask questions, inspect citations, and download authorized sources.
Only owners may list/manage members. Membership revocation is effective on subsequent requests.
Workspaces have separate SQLite libraries, Qdrant directories, retrieval corpora, answer caches,
and trace stores. Only model weights are shared. Cached results are authorized before lookup.
Neither bundled benchmark documents nor the former shared library enter a workspace implicitly.

## Document lifecycle and queue

### Reading and answering

All questions go through document retrieval. Explicit code-example and ordered-procedure
requests can return verified source excerpts; supported quantity × unit-price requests use
Decimal arithmetic with recorded operands. Other requests use generation. These routes
operate on retrieved content and do not map individual questions to prewritten answers. Dense and lexical retrieval find candidates, and the cross-encoder ranks them. Its
raw scores are not calibrated probabilities: a relevant passage can score below zero. The
old `0.8` pre-generation cutoff is therefore disabled by default. Remove an existing
`PAKE_RAG_MIN_RETRIEVAL_SCORE=0.8` override to adopt this behavior; set a cutoff only after
calibrating it on your own corpus.

Workspace retrieval uses Unicode/diacritic-normalized lexical tokens, document names, and
inherited Markdown section headings. Document-name tokens provide a recall hint without excluding other documents. The console
provides explicit document selection; API clients can send `document_ids` with a question.
The server checks each selected document against the authorized workspace before retrieval
and separates selected-document cache entries from whole-workspace answers. General example, overview,
and comparison requests select distinct sections; example requests prioritize concrete
example/code passages over introduction/index sections. These categories affect retrieval; the separate excerpt route can copy requested examples
or steps directly from the selected evidence. The approach is heuristic and is not a general semantic query planner.

Selected excerpts are presented in document order, within a 1,600-token context budget and
up to 20 source blocks (still bounded by retrieval `top_k`). The workspace prompt asks the
model to read them and produce a short answer, rather than simultaneously format citations.
The application matches each generated claim to up to three supporting excerpts, requires
strong lexical coverage, matching numbers, and consistency of capitalized names/terms, then runs the existing citation validator.
Output that exhausts the generation budget is rejected or uses the existing verified extractive fallback.
Unmatched claims are rejected; absent information can still produce an abstention. This is
a lexical safeguard, not a semantic entailment guarantee. Original model output remains in
`raw_answer` for generated answers (null for calculation/excerpt routes); successful answers carry the usual source/page/version links.

The model remains Qwen2.5-0.5B-Instruct. The larger context and 320-token output ceiling can
increase uncached latency. Very short or ambiguous questions, paraphrases across languages,
and difficult summaries remain limited by this small model. Large libraries are searched
through retrieved excerpts, not exhaustively read on every question. A question such as
“who is she” has no conversation history to disambiguate multiple people across documents.
The benchmark/legacy service retains its inline-citation prompt unless explicitly constructed
with `attribute_sources=True`.

Responses also expose `retrieved_sources` separately from verified `citations`, and an
`outcome_reason` distinguishing absent evidence, an abstention, a failed verification, and
exhausted generation length. The console shows passages reviewed and verified citations
separately. When an answer fails, authorized source previews remain available and are
explicitly labeled as retrieved passages, not a verified answer.

For broad real-model regression coverage and the current quality limitations, see
[Cross-domain workspace evaluation](workspace-quality.md).

`POST /api/v1/documents?filename=...` and `PUT /api/v1/documents/{id}?filename=...`
accept bounded raw file bytes and return **202** with a `processing` record. Admission checks
(filename, extension, byte quota, document/version quotas, membership) happen immediately.
Parsing, chunking, embedding, and indexing run in a background worker. A duplicate upload
within the same workspace returns **200** without creating another job. Poll the document
list to observe `processing`, `ready`, or `failed`; failures include a safe explanation.

Jobs and source bytes are durable SQLite records. Jobs interrupted in the `running` state
are requeued on startup. Explicit retry is available for failed documents. A malformed file
can therefore be accepted at upload time and subsequently transition to `failed`.

New uploads embed only their new chunks. Replacements embed only the replacement document,
upsert those vectors, and delete the old document's vector points. Other documents' vectors
are reused. BM25 is rebuilt from active chunks in memory. Short publication operations are
serialized with workspace queries so a request cannot observe a partially replaced index.
Parsing/embedding happen before acquiring that workspace query lock. Existing ready documents
remain answerable while new uploads are processing; sharing hardware still allows resource
contention and does not promise constant latency under load.

A replacement retains the previous active version until the new one is published. Failed
parsing/indexing preserves the existing active evidence. Successful publication invalidates
that workspace's answer cache. Old successful versions remain available for old citation links.
`DELETE` removes every source version, queued job, and active chunk for that document and
invalidates cached answers. Deletion during processing wins; the job cannot resurrect it.
Previously recorded traces can retain citation metadata, but deleted source endpoints return 404.

The persisted active chunks/vectors are authoritative: startup rebuilds local Qdrant from them
without re-embedding unchanged data. Changing the embedding model/revision/dimension re-embeds
stored version chunks during preparation. Model weights are preloaded before accepting requests
by default. Run **one API process per workspaces directory**; an exclusive process lock enforces
this for the local queue and Qdrant storage. This implementation targets macOS and Linux.

### Chunk boundaries and index migration

Workspace chunks obey both generator and embedding token budgets. Long prose uses overlap;
code fences are kept whole where possible and oversized code is split into fenced fragments.
Fragments are evidence, not promised executable programs. Table fragments repeat their column
header. Atomic block boundaries can omit prose overlap; adjacent chunks in the same document,
version, page and section can be joined for context, with `supporting_chunk_ids` retaining lineage.
Very long questions and generation inputs exceeding model capacity fail explicitly instead
of silently truncating. Context can still omit evidence when its configured budget is exhausted.

Chunks store a versioned ingestion profile. On startup, ready documents with an older profile
are queued for replacement from their saved original bytes. The previous active version stays
available until publication succeeds, which invalidates cached answers. This consumes a document
version; documents already at the configured version limit are not automatically migrated.
Changing parser/chunker behavior requires bumping the profile version. The dated audit and older
chunking benchmark reports describe their original runs, not these new indexes.

## PDF, DOCX, and source inspection

- PDF uses layout extraction to improve reading order and preserves physical page numbers;
  chunks do not cross page boundaries. Complex layouts can still need manual review.
- DOCX groups body paragraphs under headings and extracts simple tables, headers and footers.
  Locations are numbered sections, not fabricated pages. Text boxes, nested tables, footnotes,
  images and embedded objects are not comprehensively supported.
- Markdown/text remain UTF-8 sources. Their preview is one source section.
- Citation responses include `document_version`, `source_unit`, and `source_kind`.
  **Open page/section** shows the extracted evidence; **Download original** retrieves the
  exact authenticated source bytes for that version.
- Image-only PDF pages use local OCR when `PAKE_UPLOAD_OCR_ENABLED=true`. Install Poppler
  (`pdftoppm`) and Tesseract on the API host. The default language is `eng`; `eng+tur` requires
  both language packs installed locally. Missing tools or failed extraction produce an actionable
  processing error. OCR quality depends on the scan and is not guaranteed. Each subprocess
  has a 15-second limit, and the whole-document parser limits below still apply; long scanned
  documents may exceed them. Blank pages retain their physical positions.
- PDF/DOCX parsing runs in a separate subprocess with a 30-second wall timeout and 20-second
  CPU limit. Linux workers also have a 512 MiB address-space limit. Extracted text, PDF page
  count, PDF page streams, and DOCX ZIP expansion have configured limits. These bounds do
  not make the parser a complete security sandbox; keep parser dependencies updated.

Defaults per workspace: 100 documents, 2,000 active chunks, 10 versions per document,
5 MiB per original file, 20 MiB expanded/extracted content, and 500 PDF pages. Retained
versions consume storage; the version cap bounds their growth. Database row deletion does
not claim forensic erasure of disk blocks or backups.

## API additions

| Method | Route | Access |
|---|---|---|
| POST | `/api/v1/auth/register` | Public when registration is enabled |
| POST | `/api/v1/auth/login` | Public; rate limited |
| POST | `/api/v1/auth/logout` | Invalidates the current session |
| GET | `/api/v1/auth/me` | Session; returns user and memberships |
| GET/POST | `/api/v1/workspace/members` | Owner |
| DELETE | `/api/v1/workspace/members/{user_id}` | Owner; owner cannot be removed here |
| GET/POST | `/api/v1/documents` | Member / editor or owner |
| PUT/DELETE | `/api/v1/documents/{document_id}` | Editor or owner |
| POST | `/api/v1/documents/{document_id}/retry` | Editor or owner |
| GET | `/api/v1/documents/{document_id}/source?version=1&unit=1` | Member |
| GET | `/api/v1/documents/{document_id}/download?version=1` | Member |
| POST | `/api/v1/answers`, `/rag/answer` | Member |
| GET | `/api/v1/traces`, `/api/v1/traces/{trace_id}` | Member |

`/health`, `/api/v1/health`, and API schema/docs remain public and contain no document data.
Other routes reject missing/expired sessions with 401, insufficient roles with 403, and
foreign workspace/document/trace IDs with 404. HTTP response caching is disabled.

## Existing shared documents

The previous `data/uploads/documents.sqlite3` is preserved. Assigning it automatically to a
company would invent ownership, so import is an explicit local administrator operation:

```bash
uv run python -m scripts.import_legacy_documents --workspace-id YOUR_WORKSPACE_ID
```

The ID is returned by `/api/v1/auth/me` or `scripts.create_account`. The importer opens the
old database read-only and queues its saved text in the selected workspace. The old system
stored normalized text rather than original file bytes, so those imported originals are the
saved text. Restart/start the API to process jobs. Repeated imports deduplicate identical text.

## Verification

Generation automatically uses OpenAI when the backend has `OPENAI_API_KEY`; otherwise
it uses the existing local model. Workspace authorization, indexing, and source
validation apply to both modes. See [generation providers](generation-providers.md)
for setup, provider errors, and which data is sent to the API.

`tests/test_workspaces.py` exercises the production app with isolated temporary stores and
small deterministic model adapters. It covers two-company answers and caches, foreign-source
and trace access, forged workspace headers, reader restrictions and revocation, password/session
storage, logout/expiry/CSRF/rate limits, incremental embedding, background processing, failed
replacement, PDF page lineage, DOCX tables, original versions, and restart recovery.
Earlier service tests continue to validate the benchmark/legacy pipeline without mounting it.
