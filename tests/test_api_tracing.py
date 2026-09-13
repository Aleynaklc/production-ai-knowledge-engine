"""Versioned API, request correlation, CORS, and bounded trace tests."""

from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from backend.app.config import Settings
from backend.app.observability.models import SystemTrace, TraceTokenUsage
from backend.app.observability.store import TraceStore
from backend.app.rag.citations import CitationValidation
from backend.app.rag.models import AnswerCitation, RAGAnswer, RAGTiming
from tests.legacy_api import create_app


class FakeAnswerService:
    """Return deterministic diagnostics without loading local ML models."""

    def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
        del top_k
        return RAGAnswer(
            question=question,
            status="answered",
            answer="Access tokens expire after 15 minutes [S1].",
            raw_answer="Access tokens expire after 15 minutes [S1].",
            fallback_used=False,
            citations=[
                AnswerCitation(
                    citation_id="S1",
                    chunk_id="chunk-1",
                    document_id="security-handbook",
                    source="security.md",
                    title="Security handbook",
                    snippet="Access tokens expire after 15 minutes.",
                    retrieval_score=4.25,
                )
            ],
            validation=CitationValidation(
                valid=True,
                abstained=False,
                cited_source_ids=["S1"],
                unknown_source_ids=[],
                uncited_claims=[],
                unsupported_claims=[],
                issues=[],
            ),
            context_source_count=1,
            context_token_count=42,
            timings=RAGTiming(
                retrieval_ms=8.5,
                context_ms=1.2,
                generation_ms=24.0,
                grounding_ms=0.8,
                total_ms=35.1,
            ),
            input_tokens=96,
            output_tokens=10,
        )


def test_versioned_answer_returns_and_persists_correlated_trace() -> None:
    store = TraceStore(max_records=5)
    client = TestClient(
        create_app(
            Settings(environment="test"),
            rag_service=FakeAnswerService(),
            trace_store=store,
        )
    )

    response = client.post(
        "/api/v1/answers",
        json={"question": "How long do access tokens last?"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert response.headers["x-request-id"] == payload["request_id"]
    assert payload["trace_id"] == payload["trace"]["trace_id"]
    assert payload["result"]["status"] == "answered"
    assert [stage["name"] for stage in payload["trace"]["stages"]] == [
        "retrieval",
        "context_assembly",
        "generation",
        "grounding",
    ]

    stored = client.get(f"/api/v1/traces/{payload['trace_id']}")
    recent = client.get("/api/v1/traces?limit=1")
    assert stored.status_code == 200
    assert stored.json()["request_id"] == payload["request_id"]
    assert recent.json()["count"] == 1
    assert recent.json()["traces"][0]["trace_id"] == payload["trace_id"]


def test_system_endpoint_describes_local_keyless_runtime() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.get("/api/v1/system")

    assert response.status_code == 200
    assert response.json()["inference"] == "local"
    assert response.json()["api_key_required"] is False
    assert response.json()["api_version"] == "1.0.0"


def test_validation_and_trace_not_found_use_stable_error_envelope() -> None:
    client = TestClient(create_app(Settings(environment="test"), rag_service=FakeAnswerService()))

    invalid = client.post("/api/v1/answers", json={"question": ""})
    missing = client.get("/api/v1/traces/does-not-exist")

    assert invalid.status_code == 422
    assert invalid.json()["error"]["code"] == "request_validation_failed"
    assert invalid.headers["x-request-id"] == invalid.json()["error"]["request_id"]
    assert missing.status_code == 404
    assert missing.json()["error"]["code"] == "trace_not_found"


def test_configured_frontend_origin_is_allowed_by_cors() -> None:
    client = TestClient(create_app(Settings(environment="test")))

    response = client.options(
        "/api/v1/answers",
        headers={
            "Origin": "http://localhost:5173",
            "Access-Control-Request-Method": "POST",
        },
    )

    assert response.status_code == 200
    assert response.headers["access-control-allow-origin"] == "http://localhost:5173"


def _trace(trace_id: str, created_at: datetime) -> SystemTrace:
    return SystemTrace(
        trace_id=trace_id,
        request_id=f"request-{trace_id}",
        created_at=created_at,
        question="question",
        status="abstained",
        total_ms=1.0,
        stages=[],
        sources=[],
        tokens=TraceTokenUsage(context=0, input=0, output=0),
        validation_issues=[],
        fallback_used=False,
    )


def test_trace_store_is_bounded_and_newest_first() -> None:
    store = TraceStore(max_records=2)
    started = datetime.now(UTC)
    store.add(_trace("one", started))
    store.add(_trace("two", started + timedelta(seconds=1)))
    store.add(_trace("three", started + timedelta(seconds=2)))

    assert store.get("one") is None
    assert [trace.trace_id for trace in store.recent()] == ["three", "two"]


def test_empty_injected_trace_store_retains_its_identity_and_capacity() -> None:
    store = TraceStore(max_records=1)
    app = create_app(rag_service=FakeAnswerService(), trace_store=store)
    assert app.state.trace_store is store
    with TestClient(app) as client:
        for question in ("first", "second"):
            assert client.post("/api/v1/answers", json={"question": question}).status_code == 200
    assert len(store) == 1
    assert store.recent()[0].question == "second"


@pytest.mark.parametrize("route", ["/api/v1/answers", "/rag/answer"])
def test_runtime_failure_returns_safe_correlated_error_and_can_recover(route: str) -> None:
    class FailingService(FakeAnswerService):
        failed = False

        def answer(self, question: str, top_k: int | None = None) -> RAGAnswer:
            if not self.failed:
                self.failed = True
                raise RuntimeError("private model path /secret/model unavailable")
            return super().answer(question, top_k)

    store = TraceStore()
    with TestClient(create_app(rag_service=FailingService(), trace_store=store)) as client:
        failed = client.post(route, json={"question": "A valid question"})
        assert failed.status_code == 503
        assert failed.json()["error"]["code"] == "rag_unavailable"
        assert failed.json()["error"]["request_id"] == failed.headers["x-request-id"]
        assert "/secret" not in failed.text
        assert len(store) == 0
        assert client.post(route, json={"question": "Retry question"}).status_code == 200
        assert len(store) == 1


@pytest.mark.parametrize("route", ["/api/v1/answers", "/rag/answer"])
def test_whitespace_question_is_rejected_before_model_work(route: str) -> None:
    with TestClient(create_app(Settings(rag_preload_on_startup=False))) as client:
        response = client.post(route, json={"question": " \n\t "})
    assert response.status_code == 422
    assert response.json()["error"]["code"] == "request_validation_failed"
