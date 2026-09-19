"""Autoregressive text generation with explicit sampling controls and metrics."""

from time import perf_counter
from typing import Literal, cast

import torch
from pydantic import BaseModel, Field

from backend.app.llm.model import ModelRuntime
from backend.app.llm.tokenizer import DEFAULT_SYSTEM_PROMPT, build_chat_inputs


class GenerationOptions(BaseModel):
    """Validated decoding controls shared by the CLI and future API."""

    temperature: float = Field(default=0.7, gt=0.0, le=2.0)
    top_k: int = Field(default=50, ge=0, le=1_000)
    top_p: float = Field(default=0.9, gt=0.0, le=1.0)
    max_new_tokens: int = Field(default=128, ge=1, le=2_048)
    repetition_penalty: float = Field(default=1.0, gt=0.0, le=2.0)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    do_sample: bool = True


class ContextWindowExceeded(ValueError):
    """The full chat template plus requested completion cannot fit the model."""


class GenerationResult(BaseModel):
    """Generated text plus reproducible performance measurements."""

    text: str
    input_tokens: int
    output_tokens: int
    generation_seconds: float
    tokens_per_second: float
    options: GenerationOptions
    provider: Literal["local", "openai"] = "local"
    model: str | None = None


def build_generation_kwargs(
    options: GenerationOptions,
    *,
    pad_token_id: int | None,
    eos_token_id: int | list[int] | None,
) -> dict[str, object]:
    """Translate validated options into warning-free Transformers arguments."""

    kwargs: dict[str, object] = {
        "do_sample": options.do_sample,
        "max_new_tokens": options.max_new_tokens,
        "repetition_penalty": options.repetition_penalty,
        "use_cache": True,
    }
    if pad_token_id is not None:
        kwargs["pad_token_id"] = pad_token_id
    if eos_token_id is not None:
        kwargs["eos_token_id"] = eos_token_id
    if options.do_sample:
        kwargs.update(
            temperature=options.temperature,
            top_k=options.top_k,
            top_p=options.top_p,
        )
    return kwargs


def _synchronize(device: torch.device) -> None:
    """Wait for asynchronous accelerator work before taking a timestamp."""

    if device.type == "cuda":
        torch.cuda.synchronize(device)
    elif device.type == "mps":
        torch.mps.synchronize()


def generate_text(
    runtime: ModelRuntime,
    prompt: str,
    options: GenerationOptions,
    system_prompt: str = DEFAULT_SYSTEM_PROMPT,
) -> GenerationResult:
    """Generate one assistant response and measure decode throughput."""

    inputs = build_chat_inputs(runtime.tokenizer, prompt, system_prompt, runtime.device)
    input_token_count = int(inputs["input_ids"].shape[-1])
    context_limit = getattr(getattr(runtime.model, "config", None), "max_position_embeddings", None)
    if (
        isinstance(context_limit, int)
        and input_token_count + options.max_new_tokens > context_limit
    ):
        raise ContextWindowExceeded(
            "The question and document context exceed the model's input limit."
        )
    torch.manual_seed(options.seed)
    generation_kwargs = build_generation_kwargs(
        options,
        pad_token_id=runtime.tokenizer.pad_token_id,
        eos_token_id=runtime.tokenizer.eos_token_id,
    )

    _synchronize(runtime.device)
    started_at = perf_counter()
    call_kwargs: dict[str, object] = {**inputs, **generation_kwargs}
    with torch.inference_mode():
        sequences = runtime.model.generate(**call_kwargs)
    _synchronize(runtime.device)
    elapsed = perf_counter() - started_at

    new_token_ids = sequences[0, input_token_count:]
    output_token_count = int(new_token_ids.shape[-1])
    text = cast(
        str,
        runtime.tokenizer.decode(new_token_ids, skip_special_tokens=True),
    ).strip()
    tokens_per_second = output_token_count / elapsed if elapsed > 0 else 0.0
    return GenerationResult(
        text=text,
        input_tokens=input_token_count,
        output_tokens=output_token_count,
        generation_seconds=round(elapsed, 6),
        tokens_per_second=round(tokens_per_second, 6),
        options=options,
        model=getattr(runtime, "model_name", None),
    )
