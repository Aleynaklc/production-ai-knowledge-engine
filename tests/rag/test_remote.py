"""Offline transport contracts: provider selection, privacy, errors, and actual token usage."""

import json
from collections.abc import Callable
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import httpx
import pytest
from pydantic import SecretStr, ValidationError

from backend.app.config import Settings
from backend.app.llm.generation import ContextWindowExceeded, GenerationOptions
from backend.app.rag.generator import LocalGroundedGenerator
from backend.app.rag.remote import GenerationProviderError, OpenAIGroundedGenerator
from backend.app.workspaces.engine import LockedGenerator, SharedModels

KEY = "test-key-never-use-for-a-real-request"


def payload() -> dict[str, object]:
    return {
        "status": "completed",
        "output": [
            {"type": "reasoning", "summary": []},
            {
                "type": "message",
                "role": "assistant",
                "content": [{"type": "output_text", "text": "Access code is SILVER-44 [S1]."}],
            },
        ],
        "usage": {"input_tokens": 87, "output_tokens": 13},
    }


def generator(
    handler: Callable[[httpx.Request], httpx.Response],
) -> OpenAIGroundedGenerator:
    return OpenAIGroundedGenerator(
        SecretStr(KEY),
        "gpt-4.1-mini",
        GenerationOptions(max_new_tokens=320, do_sample=False),
        timeout_seconds=7,
        transport=httpx.MockTransport(handler),
    )


def test_request_contract_and_provider_reported_usage() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        assert str(request.url) == "https://api.openai.com/v1/responses"
        assert request.headers["authorization"] == f"Bearer {KEY}"
        assert request.extensions["timeout"]["read"] == 7
        assert json.loads(request.content) == {
            "model": "gpt-4.1-mini",
            "instructions": "Read only the supplied sources.",
            "input": "Selected document evidence",
            "max_output_tokens": 320,
            "store": False,
        }
        return httpx.Response(200, json=payload())

    result = generator(handler).generate(
        "Selected document evidence", "Read only the supplied sources."
    )
    assert result.text == "Access code is SILVER-44 [S1]."
    assert (result.input_tokens, result.output_tokens) == (87, 13)
    assert result.provider == "openai" and result.model == "gpt-4.1-mini"


@pytest.mark.parametrize(
    "status,code",
    [
        (401, "generation_auth_failed"),
        (403, "generation_auth_failed"),
        (429, "generation_rate_limited"),
        (500, "generation_request_failed"),
        (400, "generation_request_failed"),
        (404, "generation_request_failed"),
        (302, "generation_request_failed"),
    ],
)
def test_errors_are_sanitized_without_retries_or_redirects(status: int, code: str) -> None:
    calls = []

    def handler(request: httpx.Request) -> httpx.Response:
        calls.append(request)
        return httpx.Response(
            status,
            headers={"Location": "https://unrelated.invalid/"},
            json={"error": {"message": KEY + " private source text"}},
        )

    with pytest.raises(GenerationProviderError) as caught:
        generator(handler).generate("private source text", "system")
    assert caught.value.code == code
    assert KEY not in str(caught.value) and "private source" not in str(caught.value)
    assert len(calls) == 1


@pytest.mark.parametrize(
    "failure,code",
    [
        (httpx.ReadTimeout, "generation_timeout"),
        (httpx.ConnectError, "generation_unavailable"),
    ],
)
def test_transport_failures_have_safe_errors(failure: type[httpx.RequestError], code: str) -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        raise failure(KEY, request=request)

    with pytest.raises(GenerationProviderError) as caught:
        generator(handler).generate("evidence", "system")
    assert caught.value.code == code and KEY not in str(caught.value)
    assert caught.value.__suppress_context__


@pytest.mark.parametrize("status", ["incomplete", "failed", "queued", "in_progress"])
def test_partial_or_unfinished_outputs_are_never_answers(status: str) -> None:
    data = payload()
    data["status"] = status
    with pytest.raises(GenerationProviderError):
        generator(lambda _: httpx.Response(200, json=data)).generate("evidence", "system")


@pytest.mark.parametrize("body", [b"not JSON", b"{}", b'{"status":"completed","output":[]}'])
def test_malformed_or_empty_success_responses_fail_closed(body: bytes) -> None:
    with pytest.raises(GenerationProviderError, match="response|answer"):
        generator(lambda _: httpx.Response(200, content=body)).generate("evidence", "system")


def test_refusal_is_not_turned_into_an_answer() -> None:
    data = payload()
    data["output"] = [
        {
            "type": "message",
            "role": "assistant",
            "content": [{"type": "refusal", "refusal": "private"}],
        }
    ]
    with pytest.raises(GenerationProviderError) as caught:
        generator(lambda _: httpx.Response(200, json=data)).generate("evidence", "system")
    assert caught.value.code == "generation_refused"


def test_context_limit_maps_to_existing_context_error() -> None:
    response = httpx.Response(
        400, json={"error": {"code": "context_length_exceeded", "message": KEY}}
    )
    with pytest.raises(ContextWindowExceeded) as caught:
        generator(lambda _: response).generate("evidence", "system")
    assert KEY not in str(caught.value)


@pytest.fixture
def clean_keys(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    for name in ("OPENAI_API_KEY", "PAKE_OPENAI_API_KEY", "PAKE_GENERATION_PROVIDER"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.chdir(tmp_path)  # Never load the developer's backend .env in these tests.


@pytest.mark.usefixtures("clean_keys")
@pytest.mark.parametrize(
    "key,expected", [(None, "local"), ("", "local"), ("  ", "local"), (KEY, "openai")]
)
def test_automatic_selection_and_secret_redaction(key: str | None, expected: str) -> None:
    settings = Settings(openai_api_key=SecretStr(key) if key is not None else None)
    assert settings.effective_generation_provider == expected
    assert KEY not in repr(settings) and KEY not in settings.model_dump_json()
    assert "openai_api_key" not in settings.model_dump()


@pytest.mark.usefixtures("clean_keys")
@pytest.mark.parametrize("name", ["OPENAI_API_KEY", "PAKE_OPENAI_API_KEY"])
def test_key_environment_aliases(name: str, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv(name, " " + KEY + " ")
    settings = Settings()
    assert settings.openai_api_key == SecretStr(KEY)
    assert settings.effective_generation_provider == "openai"
    assert Settings(generation_provider="local").effective_generation_provider == "local"


@pytest.mark.usefixtures("clean_keys")
def test_forced_openai_requires_credentials() -> None:
    with pytest.raises(ValidationError, match="requires OPENAI_API_KEY"):
        Settings(generation_provider="openai")


@pytest.mark.usefixtures("clean_keys")
@pytest.mark.parametrize("key,expected", [("", "local"), (KEY, "openai")])
def test_dotenv_selection(tmp_path: Path, key: str, expected: str) -> None:
    env = tmp_path / ".env"
    env.write_text(f"OPENAI_API_KEY={key}\nPAKE_GENERATION_PROVIDER=auto\n")
    assert Settings().effective_generation_provider == expected


@pytest.mark.usefixtures("clean_keys")
@pytest.mark.parametrize("mode", ["local", "openai"])
def test_api_mode_never_loads_local_generator_weights(mode: str) -> None:
    settings = Settings(openai_api_key=SecretStr(KEY) if mode == "openai" else None)
    fake_tokenizer = object()
    encoder = SimpleNamespace(tokenizer=fake_tokenizer, max_content_tokens=254, dimension=2)
    runtime = SimpleNamespace(tokenizer=fake_tokenizer)
    with (
        patch(
            "backend.app.workspaces.engine.SentenceTransformerEmbedder",
            return_value=encoder,
        ),
        patch("backend.app.workspaces.engine.CrossEncoderScorer"),
        patch("backend.app.workspaces.engine.load_runtime", return_value=runtime) as local,
        patch(
            "backend.app.workspaces.engine.load_tokenizer", return_value=fake_tokenizer
        ) as tokenizer,
    ):
        shared = SharedModels.load(settings)
    assert isinstance(shared.generator, LockedGenerator)
    if mode == "openai":
        local.assert_not_called()
        tokenizer.assert_called_once()
        assert isinstance(shared.generator.base, OpenAIGroundedGenerator)
    else:
        local.assert_called_once()
        tokenizer.assert_not_called()
        assert isinstance(shared.generator.base, LocalGroundedGenerator)
