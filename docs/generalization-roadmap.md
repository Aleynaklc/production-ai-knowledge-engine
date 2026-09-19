# Generalization work — 2026-09-16

## Scope and acceptance policy

The objective is reliable answers to new questions and documents, rather than an expanding
list of recognized prompts. The existing 40 questions are a development regression suite.
A new 24-question acceptance suite covers six new domains: archives, plant propagation,
object storage, conferences, fermentation logs, and shuttle timetables. It contains 18
answerable and six deliberately unanswerable questions, including conditions, negation,
entity/attribute selection, procedures, comparisons, and Turkish questions over English text.

The acceptance JSON was written before the implementation experiments. Its SHA-256 is
recorded in `data/evaluation/workspace_acceptance_manifest.json`; tests reject changed labels
and overlapping development questions/documents. The runner accepts
`--expected-dataset-sha256` to stop before loading models if the frozen data changes.
These are synthetic cases authored by the same developer, not an external blind benchmark.
Do not tune production rules from individual acceptance failures. Once cases are used for
such tuning, move them into development coverage and create a fresh acceptance suite.

## Tested structural change — rejected for the default runtime

The candidate stopped chunks at Markdown heading boundaries, kept overlap inside sections,
and used the last heading when a title and section preceded the same body. This removed an
algorithm mix-up in one development output, but reduced acceptance success from 14/18 to 13/18.
It was rolled back. The reproducible change (including its tests and ingestion-profile version)
is retained in `evaluation/experiments/section_boundaries.patch`; it is not active runtime code.
Applying it requires a deliberate experiment and reindexing. Existing uploaded documents do
not need migration for the retained changes.

## Implemented general changes
- Number verification recognizes a cited document's filename as a document qualifier,
  without requiring that name inside every chunk. Numeric support and the requested
  attribute still require source evidence. This remains a lexical check.
- Rejected answers no longer claim that the source necessarily lacks information. API
  clients receive an explicit verification failure message; raw answers remain separate.
- Evaluation reports retain the exact rendered context, truncation/source map, runtime
  code hash, retrieval configuration, rejected/abstained counts, and raw-answer label matches.
  A raw label match is a diagnostic proxy, never proof that a rejected answer was correct.
- Evidence-span labels are validated against their labeled documents. The manual real-model
  CI workflow evaluates development and acceptance suites separately and retains failures.

A relative reranker score window was tested on the development suite. A width of 6 omitted
one required summary passage; it is **disabled by default**. `PAKE_RAG_RERANK_SCORE_GAP` is an
optional positive experimental setting. Leave it unset to disable it. Raw cross-encoder
scores are not calibrated probabilities. A score window must not be enabled merely because
it improves one code question.

## Results

| Run | Development answerable | Acceptance answerable | False missing-fact answers |
|---|---:|---:|---:|
| Before | 28/32 | 14/18 | 0/8 and 0/6 |
| Section-boundary candidate | 28/32 | 13/18 | 0/8 and 0/6 |
| Retained runtime after rollback | 28/32 | 14/18 | 0/8 and 0/6 |

The retained runtime preserves measured quality; **this iteration does not establish an
accuracy improvement**. Acceptance remains below the 80% absolute gate. Positive-case
lexical failures include one displayed answer in the acceptance set; zero false answers on
missing facts does not mean zero wrong displayed answers overall. All labeled evidence spans
were present in the retained runs. Both suites use the same offline 0.5B model on CPU. Timings
are descriptive, not a controlled performance comparison.

Reports are `evaluation/reports/generalization_{development,acceptance}_{candidate,retained}.json`.
The acceptance baseline is `evaluation/reports/acceptance_before.json`; development baseline is
`evaluation/reports/workspace_scenarios_v6.json`. An early intermediate acceptance run was not
inspected or used to tune the candidate; the final candidate was selected from development
results before inspecting its acceptance summary. No individual acceptance failure was used
for a runtime rule. The pre-existing real-document failures remain open.

Run a regression and absolute-quality comparison:

```bash
uv run python -m scripts.compare_workspace_reports \
  evaluation/reports/acceptance_before.json \
  evaluation/reports/generalization_acceptance_retained.json
```

This intentionally exits 1: matching a failing baseline is not enough to pass the absolute
gate. Different dataset hashes or question counts cannot be compared. A passing absolute
gate also cannot hide lower positive-case success, more false missing-fact answers, or lower
evidence-span coverage. Comparisons are regression proxies, not semantic judgments.

## Remaining work

1. Evaluate multilingual semantic entailment on a separate labeled verification set, with
   valid paraphrases, contradictions, swapped entities/attributes, dates, units and missing
   facts. Do not replace evidence checking with the generator approving its own answer.
2. Add ambiguity handling and a bounded evidence-search retry policy, preserving authorization,
   latency limits and explicit abstention. These are not implemented by the present changes.
3. Measure model reasoning independently from retrieval and verification, using complete
   labeled evidence as an input. The current generator remains Qwen2.5-0.5B; no weights were
   downloaded. Existing lexical guards remain limited across languages and semantic categories.
4. Add independently authored or human-reviewed acceptance cases. Keyword checks can accept
   a response that contains the right terms but states the wrong relationship. Review semantic
   correctness, completeness, citation support, and appropriate abstention separately.

Do not read success on these 64 questions as a guarantee for arbitrary documents. The earlier
private-document limitations remain open unless separately rechecked; synthetic results do
not supersede them.

## Implementation verification

Final full suite: 260 tests passed; Ruff, formatting, and mypy passed (108 Python files).
The local API starts successfully with cached offline models. No frontend source changed
in this iteration. No remote workflow was dispatched and no deployment or commit was made.
