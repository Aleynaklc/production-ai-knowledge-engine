"""FastAPI application entry point."""

from typing import Literal

from fastapi import FastAPI
from pydantic import BaseModel

from backend.app.config import Settings, get_settings


class HealthResponse(BaseModel):
    """Public health-check response."""

    status: Literal["ok"]
    service: str
    environment: str


def create_app(settings: Settings | None = None) -> FastAPI:
    """Create an application instance with explicit, testable configuration."""

    runtime_settings = settings or get_settings()
    application = FastAPI(
        title=runtime_settings.app_name,
        version="0.1.0",
        description="API for the Production AI Knowledge Engine.",
    )

    @application.get("/health", response_model=HealthResponse, tags=["health"])
    async def health() -> HealthResponse:
        """Report that the application process is ready to receive requests."""

        return HealthResponse(
            status="ok",
            service=runtime_settings.app_name,
            environment=runtime_settings.environment,
        )

    return application


app = create_app()
