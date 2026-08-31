"""FastAPI application entry point."""

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel, Field
from starlette.concurrency import run_in_threadpool

from backend.app.config import Settings, get_settings
from backend.app.rag.factory import LazyRAGService
from backend.app.rag.models import RAGAnswer
from backend.app.rag.service import AnswerService


class HealthResponse(BaseModel):
    """Public health-check response."""

    status: Literal["ok"]
    service: str
    environment: str


class RAGRequest(BaseModel):
    """Validated grounded question submitted to the local RAG pipeline."""

    question: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)


def create_app(
    settings: Settings | None = None,
    rag_service: AnswerService | None = None,
) -> FastAPI:
    """Create an application instance with explicit, testable configuration."""

    runtime_settings = settings or get_settings()
    service_provider: AnswerService = rag_service or LazyRAGService(runtime_settings)

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        yield
        if isinstance(service_provider, LazyRAGService):
            await run_in_threadpool(service_provider.close)

    application = FastAPI(
        title=runtime_settings.app_name,
        version="0.1.0",
        description="API for the Production AI Knowledge Engine.",
        lifespan=lifespan,
    )
    application.state.rag_service = service_provider

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Report that the application process is ready to receive requests."""

        return HealthResponse(
            status="ok",
            service=runtime_settings.app_name,
            environment=runtime_settings.environment,
        )

    @application.post("/rag/answer", response_model=RAGAnswer, tags=["rag"])
    async def answer_question(request: RAGRequest) -> RAGAnswer:
        """Retrieve, generate, validate citations, and return a safe answer."""

        service: AnswerService = application.state.rag_service
        return await run_in_threadpool(service.answer, request.question, request.top_k)

    return application


app = create_app()
