"""Bounded OpenAI Responses adapter; credentials and upstream error bodies stay private."""

from time import perf_counter

import httpx
from pydantic import BaseModel, Field, SecretStr, ValidationError

from backend.app.llm.generation import ContextWindowExceeded, GenerationOptions, GenerationResult


class GenerationProviderError(RuntimeError):
    """A sanitized failure safe to expose through the application's error contract."""

    def __init__(self, code: str, message: str, status_code: int = 503) -> None:
        super().__init__(message)
        self.code, self.message, self.status_code = code, message, status_code


class OutputContent(BaseModel):
    type: str
    text: str = ""


class OutputItem(BaseModel):
    type: str
    role: str | None = None
    content: list[OutputContent] = Field(default_factory=list)


class ResponseUsage(BaseModel):
    input_tokens: int = Field(ge=0, strict=True)
    output_tokens: int = Field(ge=0, strict=True)


class ModelResponse(BaseModel):
    status: str
    output: list[OutputItem]
    usage: ResponseUsage | None = None


class OpenAIGroundedGenerator:
    """Use the official HTTPS endpoint without implicit retries or local fallback."""

    def __init__(
        self,
        api_key: SecretStr,
        model: str,
        options: GenerationOptions,
        *,
        timeout_seconds: float = 30.0,
        transport: httpx.BaseTransport | None = None,
    ) -> None:
        self._api_key = api_key
        self.model, self.options = model, options
        self.timeout_seconds, self._transport = timeout_seconds, transport

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        started = perf_counter()
        try:
            # Each request owns its connection, including all failure paths. Do not
            # forward credentials across redirects or inherit an unrelated proxy.
            with httpx.Client(
                timeout=self.timeout_seconds,
                follow_redirects=False,
                trust_env=False,
                transport=self._transport,
            ) as client:
                response = client.post(
                    "https://api.openai.com/v1/responses",
                    headers={"Authorization": f"Bearer {self._api_key.get_secret_value()}"},
                    json={
                        "model": self.model,
                        "instructions": system_prompt,
                        "input": prompt,
                        "max_output_tokens": self.options.max_new_tokens,
                        "store": False,
                    },
                )
        except httpx.TimeoutException:
            raise GenerationProviderError(
                "generation_timeout", "The model provider timed out. Please try again.", 504
            ) from None
        except httpx.RequestError:
            raise GenerationProviderError(
                "generation_unavailable", "The model provider could not be reached."
            ) from None

        if response.status_code in {401, 403}:
            raise GenerationProviderError(
                "generation_auth_failed",
                "The model provider rejected the server credentials. Check the API key and access.",
            )
        if response.status_code == 429:
            raise GenerationProviderError(
                "generation_rate_limited", "The model provider's rate or quota limit was reached."
            )
        if response.status_code == 400:
            try:
                body = response.json()
                error = body.get("error", {}) if isinstance(body, dict) else {}
                code = error.get("code") if isinstance(error, dict) else None
            except ValueError:
                code = None
            if code == "context_length_exceeded":
                raise ContextWindowExceeded(
                    "The question and document context exceed the provider's input limit."
                )
        if response.status_code != 200:
            raise GenerationProviderError(
                "generation_request_failed",
                "The model provider could not complete the request. Check the configured model and provider status.",
                502,
            )
        try:
            result = ModelResponse.model_validate_json(response.content)
        except (ValueError, ValidationError):
            raise GenerationProviderError(
                "generation_invalid_response",
                "The model provider returned an invalid response.",
                502,
            ) from None
        if result.status == "incomplete":
            raise GenerationProviderError(
                "generation_incomplete",
                "The model response was incomplete; no partial answer was accepted.",
                502,
            )
        if result.status != "completed":
            raise GenerationProviderError(
                "generation_unavailable", "The model provider did not complete the response."
            )
        content = [
            part
            for item in result.output
            if item.type == "message" and item.role == "assistant"
            for part in item.content
        ]
        if any(part.type == "refusal" for part in content):
            raise GenerationProviderError(
                "generation_refused", "The model provider declined this request.", 422
            )
        answer = "\n".join(part.text for part in content if part.type == "output_text").strip()
        if not answer or result.usage is None:
            raise GenerationProviderError(
                "generation_invalid_response",
                "The model provider returned no usable answer or token usage.",
                502,
            )
        elapsed = perf_counter() - started
        return GenerationResult(
            text=answer,
            input_tokens=result.usage.input_tokens,
            output_tokens=result.usage.output_tokens,
            generation_seconds=elapsed,
            tokens_per_second=result.usage.output_tokens / elapsed if elapsed > 0 else 0,
            options=self.options,
            provider="openai",
            model=self.model,
        )
