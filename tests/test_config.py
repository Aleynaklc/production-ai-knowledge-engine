"""Tests for application configuration."""

import pytest
from pydantic import ValidationError
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


@pytest.mark.parametrize("overlap", [32, 33])
def test_invalid_chunk_overlap_fails_at_configuration_time(overlap: int) -> None:
    with pytest.raises(ValidationError, match="smaller than chunk_size_tokens"):
        Settings(chunk_size_tokens=32, chunk_overlap_tokens=overlap)
