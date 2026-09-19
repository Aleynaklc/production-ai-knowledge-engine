"""Run synthetic, isolated cross-domain scenarios with real local models."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any
from unittest.mock import patch
from uuid import NAMESPACE_URL, uuid5

import torch

from backend.app.config import Settings
from backend.app.evaluation.workspace import read_scenarios, score_answer, summarize
from backend.app.rag.context import ContextBuilder, ContextBundle
from backend.app.rag.fallback import ExtractiveFallback
from backend.app.rag.service import RAGService
from backend.app.retrieval.hybrid import HybridRetriever
from backend.app.retrieval.models import RetrievalResult
from backend.app.retrieval.reranker import RerankedRetriever
from backend.app.retrieval.sparse import BM25Retriever
from backend.app.workspaces.engine import SharedModels, WorkspaceEngine
from backend.app.workspaces.store import WorkspaceStore

ROOT = Path(__file__).resolve().parents[1]


class RecordingContextBuilder(ContextBuilder):
    """Record the exact rendered evidence used by this evaluation's real service."""

    def __init__(self, base: ContextBuilder) -> None:
        super().__init__(
            base.codec,
            token_budget=base.token_budget,
            max_sources=base.max_sources,
            minimum_content_tokens=base.minimum_content_tokens,
            preserve_document_order=base.preserve_document_order,
            reading_format=base.reading_format,
        )
        self.last_context: ContextBundle | None = None

    def build(self, query: str, results: list[RetrievalResult]) -> ContextBundle:
        self.last_context = super().build(query, results)
        return self.last_context


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset", type=Path, default=ROOT / "data/evaluation/workspace_scenarios.json"
    )
    parser.add_argument(
        "--output", type=Path, default=ROOT / "evaluation/reports/workspace_current.json"
    )
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default="cpu")
    parser.add_argument("--variant", choices=["baseline", "current", "both"], default="both")
    parser.add_argument(
        "--enforce", action="store_true", help="Exit nonzero if the current quality gate fails"
    )
    parser.add_argument(
        "--expected-dataset-sha256", help="Fail before model loading if labels changed"
    )
    args = parser.parse_args()
    dataset_hash = hashlib.sha256(args.dataset.read_bytes()).hexdigest()
    if args.expected_dataset_sha256 and args.expected_dataset_sha256 != dataset_hash:
        parser.error("Dataset hash differs from the frozen acceptance manifest")
    dataset = read_scenarios(args.dataset)
    torch.set_num_threads(4)
    # This offline/local benchmark must not start billed calls when a server key exists.
    settings = Settings(
        device=args.device, retrieval_device=args.device, generation_provider="local"
    )
    runtime_code_hash = hashlib.sha256(
        b"".join(
            str(path.relative_to(ROOT)).encode() + b"\0" + path.read_bytes()
            for path in sorted((ROOT / "backend/app").rglob("*.py"))
        )
    ).hexdigest()
    models = SharedModels.load(settings)
    variants = ["baseline", "current"] if args.variant == "both" else [args.variant]
    rows: dict[str, list[dict[str, Any]]] = {variant: [] for variant in variants}
    with TemporaryDirectory(prefix="pake-evaluation-") as directory:
        for scenario in dataset.scenarios:
            store = WorkspaceStore(Path(directory) / scenario.id / "library.sqlite3", settings)
            engine = WorkspaceEngine(store, models, settings)
            try:
                document_ids = {}
                for document in scenario.documents:
                    # Stabilize only fixture IDs; ingestion, retrieval and models remain real.
                    with patch(
                        "backend.app.workspaces.store.uuid4",
                        return_value=uuid5(
                            NAMESPACE_URL,
                            f"pake-evaluation/{scenario.id}/{document.id}",
                        ),
                    ):
                        record, _ = store.enqueue(document.filename, document.text.encode())
                    document_ids[record.id] = document.id
                    job = store.next_job()
                    assert job is not None
                    engine.process(job)
                    if store.get(record.id).status != "ready":
                        raise RuntimeError(
                            f"Scenario ingestion failed: {scenario.id}/{document.id}"
                        )
                _, chunks, _ = store.snapshot()
                services = {"current": engine.service}
                if "baseline" in variants:
                    services["baseline"] = RAGService(
                        RerankedRetriever(
                            HybridRetriever(
                                engine.dense, BM25Retriever(chunks), settings.retrieval_candidate_k
                            ),
                            models.scorer,
                            settings.retrieval_candidate_k,
                        ),
                        ContextBuilder(
                            models.codec,
                            token_budget=settings.rag_context_tokens,
                            max_sources=settings.rag_max_sources,
                            preserve_document_order=True,
                        ),
                        models.generator,
                        default_top_k=settings.rag_top_k,
                        minimum_retrieval_score=settings.rag_min_retrieval_score,
                        attribute_sources=True,
                        extractive_fallback=ExtractiveFallback(
                            models.scorer, settings.rag_extractive_fallback_score
                        ),
                    )
                for query in scenario.queries:
                    for variant in variants:
                        service = services[variant]
                        assert service is not None
                        capture = RecordingContextBuilder(service.context_builder)
                        service.context_builder = capture
                        answer = service.answer(query.question)
                        context_text = capture.last_context.rendered if capture.last_context else ""
                        row = {
                            "scenario": scenario.id,
                            "context": capture.last_context.model_dump(mode="json")
                            if capture.last_context
                            else None,
                            **score_answer(query, answer, document_ids, context_text=context_text),
                        }
                        rows[variant].append(row)
                        print(
                            f"{variant:8} {query.id:18} {answer.status:10} pass={row['passed_lexical_regression']}",
                            flush=True,
                        )
            finally:
                engine.close()
    report = {
        "schema_version": 1,
        "created_at": datetime.now(UTC).isoformat(),
        "dataset_sha256": dataset_hash,
        "runtime_code_sha256": runtime_code_hash,
        "retrieval_settings": {
            "candidate_k": settings.retrieval_candidate_k,
            "top_k": settings.rag_top_k,
            "score_gap": settings.rag_rerank_score_gap,
            "max_sources": settings.rag_max_sources,
        },
        "model": settings.model_name,
        "model_revision": settings.model_revision,
        "device": args.device,
        "context_tokens": settings.rag_context_tokens,
        "output_tokens": settings.rag_max_new_tokens,
        "notes": "Synthetic documents with stable IDs only. Baseline uses the pre-adaptation retriever with the same current generator/validator. Both variants use Unicode BM25. Sequential requests; no answer cache. Baseline runs first, so latency is descriptive, not a controlled speed comparison. Keyword/citation checks do not establish semantic correctness. Full outputs are retained for review.",
        "variants": {
            key: {"summary": summarize(value), "per_query": value} for key, value in rows.items()
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    for variant, result in report["variants"].items():
        print(variant, json.dumps(result["summary"], ensure_ascii=False))
    tested = report["variants"].get("current", report["variants"][variants[-1]])
    return int(args.enforce and not tested["summary"]["quality_gate_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
