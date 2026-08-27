"""Tests for the minimal FastAPI application."""

import asyncio

from fastapi import FastAPI
from httpx import ASGITransport, AsyncClient, Response

from backend.app.config import Settings
from backend.app.main import create_app


async def get(app: FastAPI, path: str) -> Response:
    """Issue an HTTP request directly against the ASGI application."""

    transport = ASGITransport(app=app)
    async with AsyncClient(transport=transport, base_url="http://testserver") as client:
        return await client.get(path)


def test_health_endpoint_returns_service_status() -> None:
    """The health endpoint should expose stable readiness metadata."""

    app = create_app(Settings(environment="test"))
    response = asyncio.run(get(app, "/health"))

    assert response.status_code == 200
    assert response.json() == {
        "status": "ok",
        "service": "Production AI Knowledge Engine",
        "environment": "test",
    }


def test_openapi_schema_includes_health_endpoint() -> None:
    """FastAPI should publish the initial route through OpenAPI."""

    app = create_app(Settings(environment="test"))
    response = asyncio.run(get(app, "/openapi.json"))

    assert response.status_code == 200
    assert "/health" in response.json()["paths"]
