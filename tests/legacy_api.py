"""Historical API harness for isolated legacy service regression tests; never served."""

import logging
import sqlite3
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from typing import Literal
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool
from starlette.responses import Response

from backend.app.config import Settings, get_settings
from backend.app.documents.library import DocumentLibrary, DocumentUploadError
from backend.app.documents.models import DocumentRecord, UploadResult
from backend.app.observability.models import SystemTrace
from backend.app.observability.store import TraceStore
from backend.app.observability.tracing import build_system_trace
from backend.app.rag.factory import LazyRAGService
from backend.app.rag.models import RAGAnswer
from backend.app.rag.service import AnswerService

API_PREFIX = "/api/v1"
API_VERSION = "1.0.0"
logger = logging.getLogger(__name__)


class HealthResponse(BaseModel):
    """Public health-check response."""

    status: Literal["ok"]
    service: str
    environment: str
    api_version: str


class LegacyHealthResponse(BaseModel):
    """Stage 0 health contract retained for existing integrations."""

    status: Literal["ok"]
    service: str
    environment: str


class SystemResponse(BaseModel):
    """Safe runtime capabilities used by the frontend status panel."""

    service: str
    environment: str
    api_version: str
    inference: Literal["local"]
    model: str
    retrieval: str
    api_key_required: Literal[False]
    trace_retention: int = Field(ge=1)


class RAGRequest(BaseModel):
    """Validated grounded question submitted to the local RAG pipeline."""

    question: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)

    @field_validator("question")
    @classmethod
    def normalize_question(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("Question must contain text")
        return value


class AnswerResponse(BaseModel):
    """Grounded answer correlated with its request and system trace."""

    request_id: str
    trace_id: str
    result: RAGAnswer
    trace: SystemTrace


class TraceListResponse(BaseModel):
    """Newest-first bounded trace collection."""

    traces: list[SystemTrace]
    count: int = Field(ge=0)


class DocumentListResponse(BaseModel):
    """The shared uploaded library and the limits needed by its upload form."""

    documents: list[DocumentRecord]
    count: int
    max_file_bytes: int
    supported_extensions: list[str] = [".md", ".txt"]
    scope: Literal["shared"] = "shared"


class ErrorDetail(BaseModel):
    """Stable API error payload that never exposes internal exceptions."""

    code: str
    message: str
    request_id: str


class ErrorResponse(BaseModel):
    """Envelope for all application-owned API errors."""

    error: ErrorDetail


def _error(status_code: int, code: str, message: str, request_id: str) -> JSONResponse:
    payload = ErrorResponse(error=ErrorDetail(code=code, message=message, request_id=request_id))
    return JSONResponse(status_code=status_code, content=payload.model_dump())


def _request_id(request: Request) -> str:
    value = getattr(request.state, "request_id", None)
    return value if isinstance(value, str) else str(uuid4())


def create_app(
    settings: Settings | None = None,
    rag_service: AnswerService | None = None,
    trace_store: TraceStore | None = None,
    document_library: DocumentLibrary | None = None,
) -> FastAPI:
    """Create an application instance with explicit, testable dependencies."""

    runtime_settings = settings or get_settings()
    library = document_library or DocumentLibrary(runtime_settings)
    service_provider: AnswerService = rag_service or LazyRAGService(
        runtime_settings, document_library=library
    )
    runtime_trace_store = (
        trace_store if trace_store is not None else TraceStore(runtime_settings.trace_max_records)
    )

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            if (
                isinstance(service_provider, LazyRAGService)
                and runtime_settings.rag_preload_on_startup
            ):
                logger.info(
                    "Preparing local RAG models and retrieval index before serving requests"
                )
                await run_in_threadpool(service_provider.prepare)
                logger.info("Local RAG runtime is ready")
            yield
        finally:
            if isinstance(service_provider, LazyRAGService):
                await run_in_threadpool(service_provider.close)

    application = FastAPI(
        title=runtime_settings.app_name,
        version=API_VERSION,
        description="Versioned API for the Production AI Knowledge Engine.",
        lifespan=lifespan,
    )
    application.state.rag_service = service_provider
    application.state.trace_store = runtime_trace_store
    application.state.document_library = library
    application.add_middleware(
        CORSMiddleware,
        allow_origins=runtime_settings.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST", "OPTIONS"],
        allow_headers=["Content-Type", "X-Request-ID"],
        expose_headers=["X-Request-ID"],
    )

    @application.middleware("http")
    async def request_context(
        request: Request,
        call_next: Callable[[Request], Awaitable[Response]],
    ) -> Response:
        request.state.request_id = str(uuid4())
        response = await call_next(request)
        response.headers["X-Request-ID"] = request.state.request_id
        return response

    @application.exception_handler(RequestValidationError)
    async def validation_error(request: Request, _: RequestValidationError) -> JSONResponse:
        return _error(
            422,
            "request_validation_failed",
            "Request data did not match the API contract.",
            _request_id(request),
        )

    @application.exception_handler(ValueError)
    async def value_error(request: Request, _: ValueError) -> JSONResponse:
        return _error(
            400,
            "invalid_request",
            "The request could not be processed.",
            _request_id(request),
        )

    @application.exception_handler(DocumentUploadError)
    async def document_error(request: Request, error: DocumentUploadError) -> JSONResponse:
        return _error(error.status_code, error.code, error.message, _request_id(request))

    @application.get(
        f"{API_PREFIX}/documents", response_model=DocumentListResponse, tags=["documents"]
    )
    async def list_documents(request: Request) -> DocumentListResponse | Response:
        try:
            records = await run_in_threadpool(library.list_documents)
        except (OSError, sqlite3.Error):
            logger.exception("Document library could not be read")
            return _error(
                503,
                "document_storage_unavailable",
                "The document library is unavailable.",
                _request_id(request),
            )
        return DocumentListResponse(
            documents=records, count=len(records), max_file_bytes=runtime_settings.upload_max_bytes
        )

    @application.post(
        f"{API_PREFIX}/documents",
        response_model=UploadResult,
        status_code=201,
        responses={
            200: {"model": UploadResult},
            400: {"model": ErrorResponse},
            409: {"model": ErrorResponse},
            413: {"model": ErrorResponse},
            415: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["documents"],
        openapi_extra={
            "requestBody": {
                "required": True,
                "content": {
                    "application/octet-stream": {"schema": {"type": "string", "format": "binary"}}
                },
            }
        },
    )
    async def upload_document(
        request: Request, filename: str = Query(min_length=1, max_length=255)
    ) -> Response:
        """Upload bounded raw UTF-8 file bytes; the filename is a query parameter.

        The browser sends a File directly, avoiding multipart buffering before size validation.
        Chunking and persistence complete before success; RAG refreshes before the next answer.
        """

        length = request.headers.get("content-length")
        if length is not None:
            if not length.isascii() or not length.isdecimal():
                return _error(
                    400,
                    "invalid_content_length",
                    "Content-Length must be nonnegative.",
                    _request_id(request),
                )
            if int(length) > runtime_settings.upload_max_bytes:
                return _error(
                    413,
                    "file_too_large",
                    "The document exceeds the upload size limit.",
                    _request_id(request),
                )
        content_type = request.headers.get("content-type", "application/octet-stream").split(";")[0]
        if content_type.strip().lower() not in {
            "application/octet-stream",
            "text/plain",
            "text/markdown",
        }:
            return _error(
                415,
                "unsupported_media_type",
                "Send the file as raw text or binary bytes.",
                _request_id(request),
            )
        content = bytearray()
        async for part in request.stream():
            if len(content) + len(part) > runtime_settings.upload_max_bytes:
                return _error(
                    413,
                    "file_too_large",
                    "The document exceeds the upload size limit.",
                    _request_id(request),
                )
            content.extend(part)
        try:
            result = await run_in_threadpool(library.upload, filename, bytes(content))
        except DocumentUploadError:
            raise
        except Exception:
            logger.exception("Document upload failed before completion")
            return _error(
                503,
                "document_processing_failed",
                "The document could not be processed. Try again.",
                _request_id(request),
            )
        return JSONResponse(
            status_code=200 if result.duplicate else 201, content=result.model_dump(mode="json")
        )

    def health_payload() -> HealthResponse:
        return HealthResponse(
            status="ok",
            service=runtime_settings.app_name,
            environment=runtime_settings.environment,
            api_version=API_VERSION,
        )

    async def execute_answer(request: Request, payload: RAGRequest) -> AnswerResponse | Response:
        service: AnswerService = application.state.rag_service
        try:
            result = await run_in_threadpool(service.answer, payload.question, payload.top_k)
        except DocumentUploadError:
            raise
        except Exception:
            logger.exception("RAG request failed")
            return _error(
                503,
                "rag_unavailable",
                "The answer service is temporarily unavailable. Please retry.",
                _request_id(request),
            )
        request_id = _request_id(request)
        trace = build_system_trace(result, request_id)
        store: TraceStore = application.state.trace_store
        store.add(trace)
        return AnswerResponse(
            request_id=request_id,
            trace_id=trace.trace_id,
            result=result,
            trace=trace,
        )

    @application.get("/health", response_model=LegacyHealthResponse, tags=["health"])
    async def legacy_health() -> LegacyHealthResponse:
        """Report readiness through the backwards-compatible Stage 0 contract."""

        payload = health_payload()
        return LegacyHealthResponse(
            status=payload.status,
            service=payload.service,
            environment=payload.environment,
        )

    @application.get(f"{API_PREFIX}/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Report process readiness and version without loading local models."""

        return health_payload()

    @application.get(f"{API_PREFIX}/system", response_model=SystemResponse, tags=["system"])
    async def system() -> SystemResponse:
        """Describe safe runtime capabilities without exposing secrets."""

        return SystemResponse(
            service=runtime_settings.app_name,
            environment=runtime_settings.environment,
            api_version=API_VERSION,
            inference="local",
            model=runtime_settings.model_name,
            retrieval="Qdrant + BM25 + RRF + cross-encoder reranking",
            api_key_required=False,
            trace_retention=runtime_trace_store.max_records,
        )

    @application.post(
        f"{API_PREFIX}/answers",
        response_model=AnswerResponse,
        responses={
            400: {"model": ErrorResponse},
            422: {"model": ErrorResponse},
            503: {"model": ErrorResponse},
        },
        tags=["answers"],
    )
    async def answer_question_v1(
        request: Request, payload: RAGRequest
    ) -> AnswerResponse | Response:
        """Retrieve, generate, ground, trace, and return one answer."""

        return await execute_answer(request, payload)

    @application.post(
        "/rag/answer",
        response_model=RAGAnswer,
        responses={503: {"model": ErrorResponse}},
        tags=["compatibility"],
    )
    async def answer_question_legacy(request: Request, payload: RAGRequest) -> RAGAnswer | Response:
        """Keep the Stage 16 answer contract while recording a Stage 19 trace."""

        response = await execute_answer(request, payload)
        if isinstance(response, Response):
            return response
        return response.result

    @application.get(
        f"{API_PREFIX}/traces",
        response_model=TraceListResponse,
        tags=["traces"],
    )
    async def list_traces(limit: int = Query(default=25, ge=1, le=100)) -> TraceListResponse:
        """Return recent traces newest first."""

        traces = runtime_trace_store.recent(limit)
        return TraceListResponse(traces=traces, count=len(traces))

    @application.get(
        f"{API_PREFIX}/traces/{{trace_id}}",
        response_model=SystemTrace,
        responses={404: {"model": ErrorResponse}},
        tags=["traces"],
    )
    async def get_trace(request: Request, trace_id: str) -> SystemTrace | Response:
        """Return one trace or a stable not-found error."""

        trace = runtime_trace_store.get(trace_id)
        if trace is None:
            return _error(
                404,
                "trace_not_found",
                "No trace exists for that id.",
                _request_id(request),
            )
        return trace

    return application
