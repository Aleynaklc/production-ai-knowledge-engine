"""Tests for application configuration."""

from pytest import MonkeyPatch

from backend.app.config import Settings


def test_settings_use_prefixed_environment_variables(monkeypatch: MonkeyPatch) -> None:
    """Environment overrides must use the documented project prefix."""

    monkeypatch.setenv("PAKE_ENVIRONMENT", "test")

    settings = Settings()

    assert settings.environment == "test"
