"""Repeated, auditable local generation measurements for Stage 26."""

import hashlib
import math
import platform
import random
from collections.abc import Callable
from datetime import UTC, datetime
from time import perf_counter

import numpy as np
import torch
import transformers
from pydantic import BaseModel, ConfigDict, Field, field_validator

from backend.app.llm.generation import GenerationOptions, GenerationResult, generate_text
from backend.app.llm.model import ModelRuntime
from backend.app.llm.tokenizer import DEFAULT_SYSTEM_PROMPT

GenerateFunction = Callable[[ModelRuntime, str, GenerationOptions, str], GenerationResult]


class GenerationBenchmarkConfig(BaseModel):
    """Controls for repeat count, discarded warmups, and reproducible sampling."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    repeats: int = Field(default=3, ge=2, le=100)
    warmup_runs: int = Field(default=1, ge=0, le=20)
    seed: int = Field(default=42, ge=0, le=4_294_967_295)
    do_sample: bool = False
    system_prompt: str = DEFAULT_SYSTEM_PROMPT


class GenerationProfile(BaseModel):
    """A fixed input prompt and output-token cap, both saved with the report."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    name: str = Field(min_length=1, pattern=r"^[a-zA-Z0-9_-]+$")
    prompt: str = Field(min_length=1)
    max_new_tokens: int = Field(ge=1, le=2_048)

    @field_validator("prompt")
    @classmethod
    def validate_prompt(cls, value: str) -> str:
        """Reject whitespace-only prompts before model inference."""

        if not value.strip():
            raise ValueError("Benchmark prompts must contain non-whitespace text")
        return value


class GenerationTrial(BaseModel):
    """One measured run; discarded warmups never appear in this collection."""

    profile: str
    repetition: int
    execution_order: int
    request_seconds: float
    result: GenerationResult


class GenerationProfileSummary(BaseModel):
    """Latency percentiles and throughput using actual generated token counts."""

    profile: str
    measured_runs: int
    input_tokens_min: int
    input_tokens_max: int
    output_tokens_min: int
    output_tokens_max: int
    mean_output_tokens: float
    generation_seconds_p50: float
    generation_seconds_p95: float
    request_seconds_p50: float
    request_seconds_p95: float
    output_tokens_per_second: float
    output_cap_hit_rate: float


class GenerationBenchmarkReport(BaseModel):
    """Machine-readable report with raw outputs, configuration, and timing boundaries."""

    schema_version: int = 1
    benchmark: str = "generation_latency"
    created_at: str
    model: dict[str, str | int | float | None]
    environment: dict[str, str | int]
    config: GenerationBenchmarkConfig
    profiles: list[GenerationProfile]
    prompt_sha256: dict[str, str]
    model_load_seconds: float
    model_load_timing: str
    warmup_runs_discarded: int
    measurement_notes: list[str]
    ttft_seconds: None = None
    ttft_unavailable_reason: str = (
        "The existing non-streaming generate_text call has no first-token timestamp. "
        "Full generation latency is not a TTFT estimate."
    )
    summaries: list[GenerationProfileSummary]
    trials: list[GenerationTrial]


def build_generation_profiles(
    output_lengths: list[int] | None = None,
    prompt_lengths: list[str] | None = None,
) -> list[GenerationProfile]:
    """Cross deterministic short/medium/long prompts with requested output caps."""

    caps = [32, 64] if output_lengths is None else output_lengths
    lengths = ["short", "medium", "long"] if prompt_lengths is None else prompt_lengths
    if not caps or len(caps) != len(set(caps)):
        raise ValueError("Output lengths must be nonempty and unique")
    if not lengths or len(lengths) != len(set(lengths)):
        raise ValueError("Prompt lengths must be nonempty and unique")
    context_repeats = {"short": 1, "medium": 8, "long": 24}
    if any(length not in context_repeats for length in lengths):
        raise ValueError("Prompt lengths must be short, medium, or long")
    context = (
        "The knowledge service indexes internal documents. It retrieves supporting "
        "passages before answering. Answers include source citations. If the evidence "
        "does not support an answer, the service abstains."
    )
    profiles = []
    for length in lengths:
        prompt = "\n".join(
            f"Reference {index + 1}: {context}" for index in range(context_repeats[length])
        )
        prompt += (
            "\nExplain the service's retrieval, grounded answering, citations, and abstention "
            "behavior in detail. Use complete sentences."
        )
        for cap in caps:
            profiles.append(
                GenerationProfile(name=f"{length}_output_{cap}", prompt=prompt, max_new_tokens=cap)
            )
    return profiles


def synchronize_runtime(runtime: ModelRuntime) -> None:
    """Finish queued model transfers before the CLI stops its model-load clock."""

    if runtime.device.type == "cuda":
        torch.cuda.synchronize(runtime.device)
    elif runtime.device.type == "mps":
        torch.mps.synchronize()


def _environment(runtime: ModelRuntime) -> dict[str, str | int]:
    hardware = platform.processor() or platform.machine()
    if runtime.device.type == "cuda":
        hardware = torch.cuda.get_device_name(runtime.device)
    return {
        "platform": platform.platform(),
        "machine": platform.machine(),
        "processor_or_accelerator": hardware,
        "python": platform.python_version(),
        "torch": str(torch.__version__),
        "transformers": transformers.__version__,
        "torch_num_threads": torch.get_num_threads(),
        "torch_num_interop_threads": torch.get_num_interop_threads(),
    }


def _validate_measurement(result: GenerationResult, profile: GenerationProfile) -> None:
    if not math.isfinite(result.generation_seconds) or result.generation_seconds <= 0:
        raise ValueError("Generation timing must be positive and finite")
    if result.input_tokens < 1 or not 0 <= result.output_tokens <= profile.max_new_tokens:
        raise ValueError("Measured token counts must respect the input and output bounds")


def summarize_generation_trials(
    profile: GenerationProfile, trials: list[GenerationTrial]
) -> GenerationProfileSummary:
    """Use linear-interpolated percentiles and ratio-of-sums output throughput."""

    matching = [trial for trial in trials if trial.profile == profile.name]
    if not matching:
        raise ValueError(f"No measured trials for {profile.name}")
    durations = [trial.result.generation_seconds for trial in matching]
    requests = [trial.request_seconds for trial in matching]
    inputs = [trial.result.input_tokens for trial in matching]
    outputs = [trial.result.output_tokens for trial in matching]
    return GenerationProfileSummary(
        profile=profile.name,
        measured_runs=len(matching),
        input_tokens_min=min(inputs),
        input_tokens_max=max(inputs),
        output_tokens_min=min(outputs),
        output_tokens_max=max(outputs),
        mean_output_tokens=round(float(np.mean(outputs)), 6),
        generation_seconds_p50=round(float(np.percentile(durations, 50)), 6),
        generation_seconds_p95=round(float(np.percentile(durations, 95)), 6),
        request_seconds_p50=round(float(np.percentile(requests, 50)), 6),
        request_seconds_p95=round(float(np.percentile(requests, 95)), 6),
        output_tokens_per_second=round(sum(outputs) / sum(durations), 6),
        output_cap_hit_rate=round(
            sum(tokens == profile.max_new_tokens for tokens in outputs) / len(outputs), 6
        ),
    )


def run_generation_benchmark(
    runtime: ModelRuntime,
    profiles: list[GenerationProfile],
    config: GenerationBenchmarkConfig,
    *,
    model_load_seconds: float | None = None,
    generate: GenerateFunction = generate_text,
    clock: Callable[[], float] = perf_counter,
) -> GenerationBenchmarkReport:
    """Discard profile warmups, then measure seeded, interleaved single-request trials."""

    if not profiles or len({profile.name for profile in profiles}) != len(profiles):
        raise ValueError("Generation profiles must be nonempty and have unique names")
    load_seconds = runtime.load_seconds if model_load_seconds is None else model_load_seconds
    if not math.isfinite(load_seconds) or load_seconds < 0:
        raise ValueError("Model load time must be nonnegative and finite")

    def options_for(profile: GenerationProfile, iteration: int) -> GenerationOptions:
        return GenerationOptions(
            max_new_tokens=profile.max_new_tokens,
            seed=(config.seed + iteration) % 4_294_967_296,
            do_sample=config.do_sample,
        )

    for profile in profiles:
        for warmup in range(config.warmup_runs):
            result = generate(
                runtime, profile.prompt, options_for(profile, warmup), config.system_prompt
            )
            _validate_measurement(result, profile)

    trials: list[GenerationTrial] = []
    randomizer = random.Random(config.seed)
    for repetition in range(config.repeats):
        order = list(profiles)
        randomizer.shuffle(order)
        for profile in order:
            options = options_for(profile, repetition)
            started_at = clock()
            result = generate(runtime, profile.prompt, options, config.system_prompt)
            request_seconds = clock() - started_at
            _validate_measurement(result, profile)
            if not math.isfinite(request_seconds) or request_seconds < 0:
                raise ValueError("Request timing must be nonnegative and finite")
            trials.append(
                GenerationTrial(
                    profile=profile.name,
                    repetition=repetition,
                    execution_order=len(trials),
                    request_seconds=round(request_seconds, 6),
                    result=result,
                )
            )

    return GenerationBenchmarkReport(
        created_at=datetime.now(UTC).isoformat(),
        model=runtime.metadata(),
        environment=_environment(runtime),
        config=config,
        profiles=profiles,
        prompt_sha256={
            profile.name: hashlib.sha256(profile.prompt.encode("utf-8")).hexdigest()
            for profile in profiles
        },
        model_load_seconds=round(load_seconds, 6),
        model_load_timing=(
            "CLI wall time for tokenizer/model loading and final device synchronization; "
            "may include downloads and cache access"
            if model_load_seconds is not None
            else "Runtime loader timing; final accelerator synchronization is not guaranteed"
        ),
        warmup_runs_discarded=len(profiles) * config.warmup_runs,
        measurement_notes=[
            "Generation timing brackets synchronized model.generate, including prefill and "
            "autoregressive decoding; it excludes prompt tokenization and output text decoding.",
            "Request timing brackets generate_text, including tokenization, device transfer, "
            "generation, and output text decoding. Model loading and warmups are excluded.",
            "Output tokens/sec is sum(actual output tokens) / sum(generation seconds) per profile. "
            "The output cap is a maximum; EOS may end generation earlier, and token counts "
            "include generated special tokens.",
            "Percentiles use NumPy linear interpolation. Small repeat counts provide smoke "
            "measurements, not reliable production tail-latency estimates.",
            "Trials are single-request, batch size one, with seeded shuffled profile order in "
            "each repetition. Seed equals base seed plus repetition modulo 2**32. Greedy is "
            "the default; fixed seeds do not guarantee bitwise determinism across hardware.",
        ],
        summaries=[summarize_generation_trials(profile, trials) for profile in profiles],
        trials=trials,
    )


def render_generation_benchmark(report: GenerationBenchmarkReport) -> str:
    """Render compact, measured evidence without making production latency claims."""

    lines = [
        "# Generation Latency Benchmark — Stage 26",
        "",
        f"Measured at: `{report.created_at}`",
        "",
        f"Model: `{report.model['model_name']}`; resolved revision: "
        f"`{report.model['resolved_revision']}`; requested revision: "
        f"`{report.model['requested_revision']}`.",
        f"Device/dtype: `{report.model['device']}` / `{report.model['dtype']}`. "
        f"Model load: **{report.model_load_seconds:.3f} s**, reported separately.",
        f"{report.config.repeats} measured repetitions per profile; "
        f"{report.warmup_runs_discarded} total warmup runs discarded. "
        f"Base seed: `{report.config.seed}`; sampling: `{report.config.do_sample}`.",
        "",
        "| Profile | Input tokens | Actual output tokens | Generation p50 (s) | "
        "Generation p95 (s) | Request p50 (s) | Request p95 (s) | Output tok/s | Cap hit rate |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in report.summaries:
        lines.append(
            f"| {row.profile} | {row.input_tokens_min}–{row.input_tokens_max} | "
            f"{row.output_tokens_min}–{row.output_tokens_max} | "
            f"{row.generation_seconds_p50:.3f} | {row.generation_seconds_p95:.3f} | "
            f"{row.request_seconds_p50:.3f} | {row.request_seconds_p95:.3f} | "
            f"{row.output_tokens_per_second:.3f} | {row.output_cap_hit_rate:.1%} |"
        )
    lines.extend(["", "TTFT: **not measured**. " + report.ttft_unavailable_reason, ""])
    lines.extend(f"- {note}" for note in report.measurement_notes)
    lines.extend(["", f"Model-load boundary: {report.model_load_timing}.", "", "Environment:", ""])
    lines.extend(f"- {key}: `{value}`" for key, value in report.environment.items())
    lines.extend(
        [
            "",
            "The companion JSON contains the exact prompts, their SHA-256 hashes, every "
            "measured output, per-trial options/seeds, execution order, and model metadata.",
            "",
        ]
    )
    return "\n".join(lines)
