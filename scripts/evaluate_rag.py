"""Run and document the Stage 13–16 grounded RAG benchmark."""

import json
from pathlib import Path

from backend.app.config import Settings
from backend.app.evaluation.rag import evaluate_rag, read_rag_queries, write_rag_evaluation
from backend.app.rag.factory import build_rag_runtime

PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _write_markdown(report_path: Path, evaluation_path: Path) -> None:
    report = json.loads(evaluation_path.read_text(encoding="utf-8"))
    metrics = report["metrics"]
    failures = [
        outcome["query_id"]
        for outcome in report["per_query"]
        if outcome["answerable"] and not outcome["answer_terms_present"]
    ]
    unsafe = [
        outcome["query_id"]
        for outcome in report["per_query"]
        if not outcome["answerable"] and outcome["status"] == "answered"
    ]
    rejected = [
        outcome["query_id"] for outcome in report["per_query"] if outcome["status"] == "rejected"
    ]
    markdown = f"""# Grounded RAG Evaluation

This report is generated from 20 versioned questions: 16 answerable and 4 deliberately
unanswerable. Required answer terms provide a small deterministic correctness check;
source labels measure citation precision and recall. All generation uses the pinned local
model with greedy decoding.

Citation precision is averaged over answerable responses that contain citations. Citation
recall includes every answerable query, so abstentions reduce recall but not precision.

| Metric | Value |
|---|---:|
| Answer correctness | {metrics["answer_correctness"]:.3f} |
| Citation precision | {metrics["citation_precision"]:.3f} |
| Citation recall | {metrics["citation_recall"]:.3f} |
| Grounded answer rate | {metrics["grounded_answer_rate"]:.3f} |
| Answerable response rate | {metrics["answerable_response_rate"]:.3f} |
| Abstention accuracy | {metrics["abstention_accuracy"]:.3f} |
| Safe unanswerable rate | {metrics["safe_unanswerable_rate"]:.3f} |
| Rejection rate | {metrics["rejection_rate"]:.3f} |
| Extractive fallback rate | {metrics["extractive_fallback_rate"]:.3f} |
| Mean end-to-end latency | {metrics["mean_latency_ms"]:.2f} ms |
| p95 end-to-end latency | {metrics["p95_latency_ms"]:.2f} ms |

## Safety behavior

Answers without citations, with invented source identifiers, or with uncited/unsupported
factual claims are not returned to the caller. Strict mode first attempts a high-confidence
verbatim source fallback and otherwise returns the standard insufficient-context response.
It retains `raw_answer` and validation issues for debugging.

## Diagnostics

- Incorrect or suppressed answerable queries: `{failures}`
- Unsafe answers to unanswerable queries: `{unsafe}`
- Outputs rejected by the grounding gate: `{rejected}`

Detailed answers, citations, timings, and failure reasons are stored in
`evaluation/reports/grounded_rag_v1.json`.
"""
    report_path.write_text(markdown, encoding="utf-8")


def main() -> None:
    settings = Settings()
    queries = read_rag_queries(PROJECT_ROOT / "data/evaluation/rag_queries.jsonl")
    runtime = build_rag_runtime(settings)
    try:
        evaluation = evaluate_rag(runtime.service, queries)
    finally:
        runtime.close()

    evaluation_path = PROJECT_ROOT / "evaluation/reports/grounded_rag_v1.json"
    write_rag_evaluation(evaluation_path, evaluation)
    report_path = PROJECT_ROOT / "docs/grounded-rag-evaluation.md"
    _write_markdown(report_path, evaluation_path)
    print(json.dumps(evaluation.metrics.model_dump(mode="json"), indent=2))
    print(f"Wrote {evaluation_path}")


if __name__ == "__main__":
    main()
