"""Typed application configuration loaded from environment variables."""

from functools import lru_cache
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    """Runtime settings for the application.

    Environment variables use the ``PAKE_`` prefix. For example,
    ``PAKE_LOG_LEVEL=DEBUG`` overrides ``log_level``.
    """

    app_name: str = "Production AI Knowledge Engine"
    environment: Literal["development", "test", "production"] = "development"
    log_level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        env_prefix="PAKE_",
        extra="ignore",
    )


@lru_cache
def get_settings() -> Settings:
    """Return one cached settings object for the process."""

    return Settings()
