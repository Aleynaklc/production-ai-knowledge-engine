"""Tests for application configuration."""

from pytest import MonkeyPatch

from backend.app.config import Settings


def test_settings_use_prefixed_environment_variables(monkeypatch: MonkeyPatch) -> None:
    """Environment overrides must use the documented project prefix."""

    monkeypatch.setenv("PAKE_ENVIRONMENT", "test")

    settings = Settings()

    assert settings.environment == "test"


def test_model_settings_are_validated(monkeypatch: MonkeyPatch) -> None:
    """Local inference settings should be configurable through the same prefix."""

    monkeypatch.setenv("PAKE_DEVICE", "cpu")
    monkeypatch.setenv("PAKE_MAX_NEW_TOKENS", "64")

    settings = Settings()

    assert settings.device == "cpu"
    assert settings.max_new_tokens == 64
