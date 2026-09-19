"""Authenticated API: every document, source, answer, and trace belongs to a membership."""

import logging
import re
import sqlite3
from collections import OrderedDict
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager
from threading import Lock
from time import monotonic
from typing import Any, Literal
from urllib.parse import quote
from uuid import uuid4

from fastapi import FastAPI, Query, Request
from fastapi.exceptions import RequestValidationError
from fastapi.middleware.cors import CORSMiddleware
from fastapi.openapi.utils import get_openapi
from fastapi.responses import JSONResponse, Response
from pydantic import BaseModel, Field, field_validator
from starlette.concurrency import run_in_threadpool

from backend.app.config import Settings, get_settings
from backend.app.documents.library import DocumentUploadError
from backend.app.observability.tracing import build_system_trace
from backend.app.workspaces.engine import WorkspaceManager

logger = logging.getLogger(__name__)
COOKIE = "pake_session"


class LoginRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    password: str = Field(min_length=1, max_length=128, repr=False)

    @field_validator("email")
    @classmethod
    def email_address(cls, value: str) -> str:
        value = value.strip().casefold()
        if not re.fullmatch(r"[^@\s]+@[^@\s]+\.[^@\s]+", value):
            raise ValueError("Invalid email")
        return value


class RegisterRequest(LoginRequest):
    password: str = Field(min_length=12, max_length=128, repr=False)
    workspace_name: str = Field(min_length=1, max_length=100)

    @field_validator("workspace_name")
    @classmethod
    def name(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Workspace name cannot be blank")
        return value.strip()


class MembershipRequest(BaseModel):
    email: str = Field(min_length=3, max_length=254)
    role: Literal["editor", "reader"]


class QuestionRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2_000)
    top_k: int | None = Field(default=None, ge=1, le=20)
    document_ids: list[str] = Field(default_factory=list, max_length=20)

    @field_validator("question")
    @classmethod
    def normalize(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("Question cannot be blank")
        return value.strip()


def create_app(
    settings: Settings | None = None, manager: WorkspaceManager | None = None
) -> FastAPI:
    settings = settings or get_settings()
    manager = manager or WorkspaceManager(settings)
    accounts = manager.accounts

    @asynccontextmanager
    async def lifespan(_: FastAPI) -> AsyncIterator[None]:
        try:
            await run_in_threadpool(manager.start)
            yield
        finally:
            await run_in_threadpool(manager.close)

    app = FastAPI(title=settings.app_name, version="1.1.0", lifespan=lifespan)
    app.state.workspace_manager = manager
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Workspace-ID"],
        expose_headers=["X-Request-ID"],
    )
    auth_attempts: OrderedDict[str, tuple[float, int]] = OrderedDict()
    attempts_lock = Lock()

    def error(request: Request, status: int, code: str, message: str) -> JSONResponse:
        return JSONResponse(
            status_code=status,
            content={
                "error": {
                    "code": code,
                    "message": message,
                    "request_id": getattr(request.state, "request_id", str(uuid4())),
                }
            },
        )

    @app.middleware("http")
    async def context(
        request: Request, next_call: Callable[[Request], Awaitable[Response]]
    ) -> Response:
        request.state.request_id = str(uuid4())
        origin = request.headers.get("origin")
        if request.method in {"POST", "PUT", "DELETE"} and (
            (origin is not None and origin not in settings.cors_origins)
            or (origin is None and request.headers.get("sec-fetch-site") == "cross-site")
        ):
            response: Response = error(request, 403, "origin_denied", "This origin is not allowed.")
        else:
            response = await next_call(request)
        response.headers["X-Request-ID"] = request.state.request_id
        response.headers["Cache-Control"] = "no-store"
        response.headers["X-Content-Type-Options"] = "nosniff"
        return response

    @app.exception_handler(DocumentUploadError)
    async def domain_error(request: Request, exc: DocumentUploadError) -> JSONResponse:
        return error(request, exc.status_code, exc.code, exc.message)

    @app.exception_handler(RequestValidationError)
    async def validation_error(request: Request, exc: RequestValidationError) -> JSONResponse:
        return error(
            request,
            422,
            "request_validation_failed",
            "Request data did not match the API contract.",
        )

    @app.exception_handler(sqlite3.Error)
    async def storage_error(request: Request, exc: sqlite3.Error) -> JSONResponse:
        logger.error("Workspace storage failure", exc_info=exc)
        return error(request, 503, "storage_unavailable", "Storage is temporarily unavailable.")

    @app.exception_handler(ValueError)
    async def invalid_stored_record(request: Request, exc: ValueError) -> JSONResponse:
        logger.error("Workspace record validation failed", exc_info=exc)
        return error(request, 503, "storage_unavailable", "Stored data could not be read.")

    def throttle(request: Request) -> None:
        address = request.client.host if request.client else "unknown"
        now = monotonic()
        with attempts_lock:
            started, count = auth_attempts.get(address, (now, 0))
            if now - started >= 60:
                started, count = now, 0
            auth_attempts[address] = (started, count + 1)
            auth_attempts.move_to_end(address)
            while len(auth_attempts) > 10_000:
                auth_attempts.popitem(last=False)
            if count >= 10:
                raise DocumentUploadError(
                    "rate_limited", "Too many sign-in attempts. Try again in a minute.", 429
                )

    def current_user(request: Request) -> dict[str, str]:
        return accounts.user(request.cookies.get(COOKIE))

    def workspace(request: Request, write: bool = False) -> tuple[str, str, str]:
        user = current_user(request)
        workspace_id = request.headers.get("x-workspace-id", "")
        role = accounts.authorize(user["id"], workspace_id, write)
        return user["id"], workspace_id, role

    def session_response(user_id: str) -> JSONResponse:
        token = accounts.new_session(user_id)
        user = accounts.user(token)
        response = JSONResponse({"user": user, "workspaces": accounts.workspaces(user_id)})
        response.set_cookie(
            COOKIE,
            token,
            httponly=True,
            secure=settings.environment == "production",
            samesite="lax",
            max_age=settings.auth_session_seconds,
            path="/",
        )
        return response

    @app.post("/api/v1/auth/register")
    async def register(request: Request, payload: RegisterRequest) -> Response:
        if not settings.auth_allow_registration:
            raise DocumentUploadError(
                "registration_disabled", "Account registration is disabled.", 403
            )
        throttle(request)
        user_id = await run_in_threadpool(
            accounts.register, payload.email, payload.password, payload.workspace_name
        )
        for item in accounts.workspaces(user_id):
            manager.store(item["id"])
        return await run_in_threadpool(session_response, user_id)

    @app.post("/api/v1/auth/login")
    async def login(request: Request, payload: LoginRequest) -> Response:
        throttle(request)
        user_id = await run_in_threadpool(accounts.login, payload.email, payload.password)
        return await run_in_threadpool(session_response, user_id)

    @app.post("/api/v1/auth/logout")
    async def logout(request: Request) -> Response:
        await run_in_threadpool(accounts.logout, request.cookies.get(COOKIE))
        response = JSONResponse({"signed_out": True})
        response.delete_cookie(COOKIE, path="/")
        return response

    @app.get("/api/v1/auth/me")
    def me(request: Request) -> dict[str, Any]:
        user = current_user(request)
        return {"user": user, "workspaces": accounts.workspaces(user["id"])}

    @app.get("/api/v1/workspace/members")
    def members(request: Request) -> dict[str, Any]:
        _, workspace_id, role = workspace(request)
        if role != "owner":
            raise DocumentUploadError("permission_denied", "Only the owner can view members.", 403)
        return {"members": accounts.members(workspace_id)}

    @app.post("/api/v1/workspace/members")
    def add_member(request: Request, payload: MembershipRequest) -> dict[str, bool]:
        user_id, workspace_id, _ = workspace(request)
        accounts.add_member(user_id, workspace_id, payload.email.strip().casefold(), payload.role)
        return {"saved": True}

    @app.delete("/api/v1/workspace/members/{user_id}")
    def remove_member(request: Request, user_id: str) -> dict[str, bool]:
        owner, workspace_id, _ = workspace(request)
        accounts.remove_member(owner, workspace_id, user_id)
        return {"removed": True}

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "service": settings.app_name, "environment": settings.environment}

    @app.get("/api/v1/health")
    def api_health() -> dict[str, str]:
        return {**health(), "api_version": "1.1.0"}

    @app.get("/api/v1/system")
    def system(request: Request) -> dict[str, Any]:
        _, workspace_id, role = workspace(request)
        return {
            "service": settings.app_name,
            "environment": settings.environment,
            "api_version": "1.1.0",
            "inference": settings.effective_generation_provider,
            "model": settings.generation_model,
            "provider_selection": settings.generation_provider,
            "retrieval": "Qdrant + BM25 + RRF + reranking",
            "api_key_required": False,
            "server_api_key_configured": settings.openai_api_key is not None,
            "authentication_required": True,
            "workspace_id": workspace_id,
            "role": role,
            "trace_retention": settings.trace_max_records,
        }

    @app.get("/api/v1/documents")
    def documents(request: Request) -> dict[str, Any]:
        _, workspace_id, role = workspace(request)
        records = manager.store(workspace_id).documents()
        return {
            "documents": records,
            "count": len(records),
            "max_file_bytes": settings.upload_max_bytes,
            "supported_extensions": [".md", ".txt", ".pdf", ".docx"],
            "scope": "workspace",
            "role": role,
        }

    async def accept_upload(
        request: Request, filename: str, document_id: str | None = None
    ) -> Response:
        _, workspace_id, _ = workspace(request, write=True)
        length = request.headers.get("content-length")
        if length is not None and (not length.isascii() or not length.isdecimal()):
            raise DocumentUploadError("invalid_content_length", "Invalid Content-Length.", 400)
        if length and int(length) > settings.upload_max_bytes:
            raise DocumentUploadError("file_too_large", "The file exceeds the upload limit.", 413)
        data = bytearray()
        async for part in request.stream():
            if len(data) + len(part) > settings.upload_max_bytes:
                raise DocumentUploadError(
                    "file_too_large", "The file exceeds the upload limit.", 413
                )
            data.extend(part)
        store = manager.store(workspace_id)
        record, duplicate = await run_in_threadpool(
            store.enqueue, filename, bytes(data), document_id
        )
        manager.wake.set()
        return JSONResponse(
            status_code=200 if duplicate else 202,
            content={"document": record.model_dump(), "duplicate": duplicate},
        )

    @app.post("/api/v1/documents", status_code=202)
    async def upload(
        request: Request, filename: str = Query(min_length=1, max_length=255)
    ) -> Response:
        return await accept_upload(request, filename)

    @app.put("/api/v1/documents/{document_id}", status_code=202)
    async def replace(
        request: Request, document_id: str, filename: str = Query(min_length=1, max_length=255)
    ) -> Response:
        return await accept_upload(request, filename, document_id)

    @app.post("/api/v1/documents/{document_id}/retry", status_code=202)
    def retry(request: Request, document_id: str) -> dict[str, Any]:
        _, workspace_id, _ = workspace(request, write=True)
        record = manager.store(workspace_id).retry(document_id)
        manager.wake.set()
        return {"document": record}

    @app.delete("/api/v1/documents/{document_id}")
    def delete(request: Request, document_id: str) -> dict[str, bool]:
        _, workspace_id, _ = workspace(request, write=True)
        manager.engine(workspace_id).delete(document_id)
        return {"deleted": True}

    @app.get("/api/v1/documents/{document_id}/source")
    def source(
        request: Request,
        document_id: str,
        version: int = Query(ge=1),
        unit: int = Query(default=1, ge=1),
    ) -> dict[str, Any]:
        _, workspace_id, _ = workspace(request)
        filename, _, units = manager.store(workspace_id).source(document_id, version)
        selected = next((item for item in units if item.number == unit), None)
        if selected is None:
            raise DocumentUploadError("source_not_found", "Source location not found.", 404)
        return {
            "filename": filename,
            "version": version,
            "unit": selected,
            "unit_count": len(units),
        }

    @app.get("/api/v1/documents/{document_id}/download")
    def download(request: Request, document_id: str, version: int = Query(ge=1)) -> Response:
        _, workspace_id, _ = workspace(request)
        filename, data, _ = manager.store(workspace_id).source(document_id, version)
        return Response(
            data,
            media_type="application/octet-stream",
            headers={"Content-Disposition": f"attachment; filename*=UTF-8''{quote(filename)}"},
        )

    def execute(request: Request, payload: QuestionRequest) -> dict[str, Any]:
        _, workspace_id, _ = workspace(request)
        try:
            engine = manager.engine(workspace_id)
            answer = engine.answer(
                payload.question, payload.top_k, document_ids=payload.document_ids
            )
        except DocumentUploadError:
            raise
        except Exception as exc:
            logger.exception("Workspace answer failed")
            raise DocumentUploadError(
                "rag_unavailable", "The answer service is temporarily unavailable.", 503
            ) from exc
        trace = build_system_trace(answer, request.state.request_id)
        engine.traces.add(trace)
        return {
            "request_id": request.state.request_id,
            "trace_id": trace.trace_id,
            "result": answer,
            "trace": trace,
        }

    @app.post("/api/v1/answers")
    def answer(request: Request, payload: QuestionRequest) -> dict[str, Any]:
        return execute(request, payload)

    @app.post("/rag/answer")
    def legacy_answer(request: Request, payload: QuestionRequest) -> Any:
        return execute(request, payload)["result"]

    @app.get("/api/v1/traces")
    def traces(request: Request, limit: int = Query(default=25, ge=1, le=100)) -> dict[str, Any]:
        _, workspace_id, _ = workspace(request)
        records = manager.engine(workspace_id).traces.recent(limit)
        return {"traces": records, "count": len(records)}

    @app.get("/api/v1/traces/{trace_id}")
    def trace(request: Request, trace_id: str) -> Any:
        _, workspace_id, _ = workspace(request)
        result = manager.engine(workspace_id).traces.get(trace_id)
        if result is None:
            raise DocumentUploadError("trace_not_found", "Trace not found.", 404)
        return result

    def schema() -> dict[str, Any]:
        if app.openapi_schema is None:
            result = get_openapi(title=settings.app_name, version="1.1.0", routes=app.routes)
            result.setdefault("components", {}).setdefault("securitySchemes", {})[
                "SessionCookie"
            ] = {
                "type": "apiKey",
                "in": "cookie",
                "name": COOKIE,
                "description": "HttpOnly session established by email/password login.",
            }
            for path, operations in result["paths"].items():
                if path in {
                    "/health",
                    "/api/v1/health",
                    "/api/v1/auth/login",
                    "/api/v1/auth/register",
                }:
                    continue
                for operation in operations.values():
                    operation["security"] = [{"SessionCookie": []}]
                    if "/auth/" not in path:
                        operation.setdefault("parameters", []).append(
                            {
                                "name": "X-Workspace-ID",
                                "in": "header",
                                "required": True,
                                "schema": {"type": "string"},
                                "description": "A workspace ID from auth/me; server membership is required.",
                            }
                        )
            app.openapi_schema = result
        return app.openapi_schema

    app.openapi = schema  # type: ignore[method-assign]
    return app
