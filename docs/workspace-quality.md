# Cross-domain workspace quality

The runtime settings do not guarantee correct answers for arbitrary documents or questions.
Model quality must be measured separately from unit/integration test success. The local
0.5B generator and lexical attribution guard are especially limited on multilingual
paraphrases, code explanations, temporal conditions, calculations, and source instructions.

## Dataset and execution

`data/evaluation/workspace_scenarios.json` contains 40 labeled questions across eight synthetic
workspaces: profile, Turkish coding notes, refund policies, operations runbook, inventory
table, dated policy versions, untrusted imported instructions, and a long equipment manual.
There are 32 answerable questions and eight deliberately unanswerable questions. No uploaded
user documents, accounts, or private data enter the dataset or committed report.

```bash
uv run python -m scripts.evaluate_workspaces --variant both
```

The runner loads real local models once, ingests the fixtures into temporary isolated
libraries, and compares the prior retriever with the current workspace retriever using
the same generation prompt and validator. Fixture IDs are stable. Reports retain prompts'
questions, raw and displayed answers, evidence, missing/forbidden terms, and timings. The
temporary libraries are deleted afterwards. Existing user workspaces are not modified.

For previously downloaded models, run offline:

```bash
HF_HUB_OFFLINE=1 TRANSFORMERS_OFFLINE=1 uv run python -m scripts.evaluate_workspaces
```

The original checked-in JSON report is `evaluation/reports/workspace_scenarios_v1.json`. The baseline
uses the pre-adaptation retriever with the same current prompt, validator and Unicode
tokenizer; it is an ablation, not a reconstruction of an old release. Baseline requests run
first, so timings are descriptive, not a controlled speed comparison.

`evaluation/reports/workspace_scenarios_v2.json` repeats all 40 questions with the same
0.5B model after numeric grounding normalization and BM25 match filtering. Numerically
identical decimal spellings such as `8` and `8.00` now match without allowing different
values or signs. BM25 candidates require an actual shared term; zero or negative scores
remain eligible when terms match, which matters in small libraries.

The current retriever scored 21/32 answerable cases in v1 and 22/32 (68.75%) in v2.
Both runs falsely answered 3/8 unanswerable cases, so neither passes the gate. The change
demonstrates an implementation improvement without changing model size; it does not solve
semantic verification. Document-level retrieval recall was 100%, which does not establish
that the correct passage or all necessary details were selected. These are historical results; the runner now writes `workspace_current.json` by default
and preserves the versioned reports for comparison.

## Quality gate

```bash
uv run python -m scripts.evaluate_workspaces --variant current --enforce
```

This exits with failure unless at least 80% of answerable cases meet the lexical/source
checks and no unanswerable case is falsely answered. A false answer fails even if the
application's own citation validator accepted it. Required numbers use token boundaries:
`4` does not count as found in `40`. Expected source coverage and forbidden terms are checked.

The **Workspace model quality** GitHub Actions workflow runs the full real-model gate on
manual dispatch and saves the report even on failure. It is intentionally separate from
fast CI, which runs the schema, metrics, selection, source-lineage and isolation tests on
every PR. The real-model workflow can download the configured model weights on its runner.
No workflow has been dispatched by this implementation.

These labels are lexical regression proxies, not semantic proof. For example, a response
can mention expected terms while omitting a procedural step or describing a fact incorrectly.
Review full outputs before expanding supported capabilities. Derived arithmetic remains an
answerable case and now exercises the constrained Decimal computation route. The runner also
measures `mean_evidence_span_recall`: labeled answer-bearing text must occur in the actual
rendered context, rather than only matching the correct document ID.

## Current measured result

`evaluation/reports/workspace_scenarios_v6.json` repeats all 40 scenarios with the same
Qwen2.5-0.5B model: **28/32 answerable cases (87.5%), zero false answers on eight missing-fact
cases**, and 100% labeled evidence-span recall. This passes the defined regression gate.
Intermediate v3–v5 reports retain the unsuccessful iterations. Dataset evidence-span labels
were added after v2; question text and positive/negative cases were retained.

The improvement combines retrieval, parsing/chunk boundaries, question-scope validation,
verbatim code/procedure excerpts, and constrained table arithmetic. Excerpt and calculation
responses skip generation and expose their provenance; this is not evidence that the LLM
itself gained those capabilities. Timings mix these routes and are descriptive, not a model
throughput benchmark. The latest run used CPU, offline cached models, and no answer cache.

Four positive cases still fail:

- `coding-2`: explaining Contains Duplicate; the generated explanation confuses algorithms.
- `runbook-4`: summary; generated claims fail source verification.
- `versions-3`: explaining conflicting dated policies; lexical attribution rejects a paraphrase.
- `long_document-3`: a Turkish interval question; the correct short number is rejected by
  the English lexical scope check.

These remain open semantic limitations. Passing 40 regression cases does not guarantee
accuracy for arbitrary documents. The guard is lexical, prompt-injection defenses are bounded,
and OCR/layout recovery needs checking on real files. No larger model was downloaded.

An earlier local experiment asking the same model to answer YES/NO about supported answers
accepted all three sampled false answers. That self-check was not added as a production
mechanism. Future changes must retain negative tests and include independently labeled cases
before expanding the claimed supported scope.

## Uploaded-document follow-up

A separate local probe re-ingested temporary copies of the existing PDF and NeetCode
Markdown; neither private source text nor the full private report is committed. The explicit
example request now returns source code excerpts, and the absent warranty question is rejected.
The university list still incorrectly includes a language school; the ambiguous identity
question and two code explanation/output questions still abstain or fail verification, including
identity with an explicit PDF scope. These real-document failures are **additional** limitations,
not covered by the 40-case synthetic success rate. The full probe remains local under
`/private/tmp/pake-fix-probe.json` and may be removed by the operating system.

## Generalization acceptance suite

The 40 scenarios above are now explicitly the development suite. The frozen 24-question
acceptance suite exposes lower performance on different documents: 14/18 positive cases,
zero false answers on six missing-fact questions. It does not pass the absolute quality gate.
The retained runtime preserves these numbers; an experimental change that reduced acceptance
success was rolled back. See [generalization results and remaining work](generalization-roadmap.md).
