"""Stage 26 contracts and arithmetic without downloading or executing a model."""

import sys
from collections.abc import Callable
from types import SimpleNamespace
from typing import cast

import pytest
import torch
from pydantic import ValidationError

import backend.app.llm.generation as generation_module
from backend.app.evaluation.generation_latency import (
    GenerationBenchmarkConfig,
    GenerationBenchmarkReport,
    GenerationProfile,
    GenerationTrial,
    build_generation_profiles,
    render_generation_benchmark,
    run_generation_benchmark,
    summarize_generation_trials,
)
from backend.app.llm.generation import GenerationOptions, GenerationResult
from backend.app.llm.model import ModelRuntime
from scripts import benchmark_generation


def fake_runtime() -> ModelRuntime:
    """Provide only the metadata boundary used by the benchmark runner."""

    metadata: dict[str, str | int | float | None] = {
        "model_name": "unit-test-fake",
        "requested_revision": "test-commit",
        "resolved_revision": "test-commit",
        "device": "cpu",
        "dtype": "float32",
        "load_seconds": 0.75,
        "parameter_count": 1,
        "parameter_bytes": 4,
    }
    return cast(
        ModelRuntime,
        SimpleNamespace(device=torch.device("cpu"), load_seconds=0.75, metadata=lambda: metadata),
    )


def fake_result(
    options: GenerationOptions, seconds: float = 2.0, output_tokens: int = 4
) -> GenerationResult:
    """Synthetic measurements exist exclusively inside unit tests."""

    return GenerationResult(
        text="test-only output",
        input_tokens=12,
        output_tokens=output_tokens,
        generation_seconds=seconds,
        tokens_per_second=999.0,
        options=options,
    )


def step_clock() -> Callable[[], float]:
    """Give each surrounding request a deterministic three-second wall duration."""

    ticks = iter(float(index * 3) for index in range(1_000))
    return lambda: next(ticks)


def test_default_matrix_has_prompt_lengths_crossed_with_output_caps() -> None:
    profiles = build_generation_profiles()

    assert len(profiles) == 6
    assert [profile.max_new_tokens for profile in profiles] == [32, 64, 32, 64, 32, 64]
    assert profiles[0].prompt == profiles[1].prompt
    assert len(profiles[0].prompt) < len(profiles[2].prompt) < len(profiles[4].prompt)


def test_warmups_excluded_and_seeds_and_order_are_reproducible() -> None:
    profiles = build_generation_profiles([8], ["short", "long"])
    config = GenerationBenchmarkConfig(repeats=3, warmup_runs=1, seed=4_294_967_295)
    calls: list[tuple[str, int, bool]] = []

    def generate(
        runtime: ModelRuntime, prompt: str, options: GenerationOptions, system_prompt: str
    ) -> GenerationResult:
        calls.append((prompt, options.seed, options.do_sample))
        # Deliberately very slow warmups must not influence any summary.
        return fake_result(options, seconds=100.0 if len(calls) <= len(profiles) else 2.0)

    report = run_generation_benchmark(
        fake_runtime(),
        profiles,
        config,
        generate=generate,
        clock=step_clock(),
        model_load_seconds=8.0,
    )

    assert len(calls) == 8
    assert len(report.trials) == 6
    assert report.warmup_runs_discarded == 2
    assert report.model_load_seconds == 8.0
    assert report.model["load_seconds"] == 0.75
    assert report.ttft_seconds is None
    for summary in report.summaries:
        assert summary.measured_runs == 3
        assert summary.generation_seconds_p50 == 2.0
        assert summary.generation_seconds_p95 == 2.0
        assert summary.request_seconds_p50 == 3.0
        assert summary.output_tokens_per_second == 2.0
        assert summary.output_cap_hit_rate == 0.0
    assert [trial.result.options.seed for trial in report.trials] == [
        4_294_967_295,
        4_294_967_295,
        0,
        0,
        1,
        1,
    ]
    assert all(not trial.result.options.do_sample for trial in report.trials)

    second = run_generation_benchmark(
        fake_runtime(), profiles, config, generate=generate, clock=step_clock()
    )
    assert [(trial.profile, trial.repetition) for trial in report.trials] == [
        (trial.profile, trial.repetition) for trial in second.trials
    ]


def test_summaries_use_actual_output_tokens_and_ratio_of_sums() -> None:
    profile = GenerationProfile(name="variable", prompt="test", max_new_tokens=8)
    options = GenerationOptions(max_new_tokens=8)
    trials = [
        GenerationTrial(
            profile="variable",
            repetition=0,
            execution_order=0,
            request_seconds=2.0,
            result=fake_result(options, seconds=1.0, output_tokens=8),
        ),
        GenerationTrial(
            profile="variable",
            repetition=1,
            execution_order=1,
            request_seconds=4.0,
            result=fake_result(options, seconds=3.0, output_tokens=2),
        ),
    ]

    summary = summarize_generation_trials(profile, trials)

    assert summary.output_tokens_per_second == 2.5  # (8 + 2) / (1 + 3)
    assert summary.generation_seconds_p50 == 2.0
    assert summary.generation_seconds_p95 == 2.9
    assert summary.request_seconds_p95 == 3.9
    assert summary.mean_output_tokens == 5.0
    assert summary.output_tokens_min == 2
    assert summary.output_tokens_max == 8
    assert summary.output_cap_hit_rate == 0.5


def test_report_roundtrip_preserves_prompts_options_and_ttft_absence() -> None:
    profiles = build_generation_profiles([8], ["short"])
    report = run_generation_benchmark(
        fake_runtime(),
        profiles,
        GenerationBenchmarkConfig(repeats=2, warmup_runs=0, do_sample=True, seed=25),
        generate=lambda runtime, prompt, options, system: fake_result(options),
        clock=step_clock(),
    )

    loaded = GenerationBenchmarkReport.model_validate_json(report.model_dump_json())
    markdown = render_generation_benchmark(loaded)

    assert loaded == report
    assert loaded.profiles[0].prompt == profiles[0].prompt
    assert len(loaded.prompt_sha256[profiles[0].name]) == 64
    assert loaded.trials[0].result.options.seed == 25
    assert loaded.trials[0].result.options.do_sample is True
    assert loaded.warmup_runs_discarded == 0
    assert "TTFT: **not measured**" in markdown
    assert "unit-test-fake" in markdown
    assert "Generation p95" in markdown
    assert "not guaranteed" in loaded.model_load_timing


def test_early_eos_counts_actual_generated_ids_including_special_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    runtime = cast(
        ModelRuntime,
        SimpleNamespace(
            **vars(fake_runtime()),
            model=SimpleNamespace(generate=lambda **kwargs: torch.tensor([[11, 12, 21, 22, 0]])),
            tokenizer=SimpleNamespace(
                pad_token_id=0, eos_token_id=0, decode=lambda *args, **kwargs: "Finished."
            ),
        ),
    )
    monkeypatch.setattr(
        generation_module,
        "build_chat_inputs",
        lambda *args: {"input_ids": torch.tensor([[11, 12]])},
    )
    ticks = iter([0.0, 2.0, 2.0, 4.0])
    monkeypatch.setattr(generation_module, "perf_counter", lambda: next(ticks))

    report = run_generation_benchmark(
        runtime,
        build_generation_profiles([8], ["short"]),
        GenerationBenchmarkConfig(repeats=2, warmup_runs=0),
        clock=step_clock(),
    )

    assert all(trial.result.input_tokens == 2 for trial in report.trials)
    assert all(trial.result.output_tokens == 3 for trial in report.trials)
    assert all(trial.result.text == "Finished." for trial in report.trials)
    assert report.summaries[0].output_tokens_per_second == 1.5
    assert report.summaries[0].output_cap_hit_rate == 0.0


@pytest.mark.parametrize(
    "config",
    [{"repeats": 1}, {"warmup_runs": -1}, {"seed": -1}, {"seed": 4_294_967_296}],
)
def test_invalid_config_rejected(config: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        GenerationBenchmarkConfig.model_validate(config)


@pytest.mark.parametrize("prompt", ["", "   ", "\n\t"])
def test_empty_or_whitespace_only_profile_prompts_fail(prompt: str) -> None:
    with pytest.raises(ValidationError):
        GenerationProfile(name="empty", prompt=prompt, max_new_tokens=8)


@pytest.mark.parametrize(
    ("caps", "lengths"),
    [
        ([], ["short"]),
        ([0], ["short"]),
        ([2049], ["short"]),
        ([8, 8], ["short"]),
        ([8], []),
        ([8], ["short", "short"]),
        ([8], ["unknown"]),
    ],
)
def test_invalid_matrix_rejected(caps: list[int], lengths: list[str]) -> None:
    with pytest.raises(ValueError):
        build_generation_profiles(caps, lengths)


def test_empty_and_duplicate_profiles_rejected_before_inference() -> None:
    profile = GenerationProfile(name="one", prompt="test", max_new_tokens=8)
    for profiles in [[], [profile, profile]]:
        with pytest.raises(ValueError, match="unique names"):
            run_generation_benchmark(fake_runtime(), profiles, GenerationBenchmarkConfig())


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("nan"), float("inf")])
def test_invalid_generation_timing_rejected(seconds: float) -> None:
    with pytest.raises(ValueError, match="positive and finite"):
        run_generation_benchmark(
            fake_runtime(),
            build_generation_profiles([8], ["short"]),
            GenerationBenchmarkConfig(repeats=2, warmup_runs=0),
            generate=lambda runtime, prompt, options, system: fake_result(options, seconds),
        )


def test_generated_tokens_cannot_exceed_output_cap() -> None:
    with pytest.raises(ValueError, match="token counts"):
        run_generation_benchmark(
            fake_runtime(),
            build_generation_profiles([8], ["short"]),
            GenerationBenchmarkConfig(repeats=2, warmup_runs=0),
            generate=lambda runtime, prompt, options, system: fake_result(options, output_tokens=9),
        )


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_invalid_request_timing_rejected(seconds: float) -> None:
    ticks = iter([0.0, seconds])
    with pytest.raises(ValueError, match="Request timing must be nonnegative and finite"):
        run_generation_benchmark(
            fake_runtime(),
            build_generation_profiles([8], ["short"]),
            GenerationBenchmarkConfig(repeats=2, warmup_runs=0),
            generate=lambda runtime, prompt, options, system: fake_result(options),
            clock=lambda: next(ticks),
        )


@pytest.mark.parametrize("seconds", [-1.0, float("nan"), float("inf")])
def test_invalid_load_timing_rejected_before_inference(seconds: float) -> None:
    def fail_generate(
        runtime: ModelRuntime, prompt: str, options: GenerationOptions, system_prompt: str
    ) -> GenerationResult:
        raise AssertionError("Invalid model load time must fail before inference")

    with pytest.raises(ValueError, match="Model load time must be nonnegative and finite"):
        run_generation_benchmark(
            fake_runtime(),
            build_generation_profiles([8], ["short"]),
            GenerationBenchmarkConfig(),
            generate=fail_generate,
            model_load_seconds=seconds,
        )


@pytest.mark.parametrize(
    "arguments",
    [
        ["--repeats", "1"],
        ["--output-lengths", "0"],
        ["--output", "same.json", "--markdown-output", "same.json"],
    ],
)
def test_cli_rejects_invalid_input_before_model_loading(
    monkeypatch: pytest.MonkeyPatch, arguments: list[str]
) -> None:
    monkeypatch.setattr(sys, "argv", ["benchmark_generation", *arguments])

    def fail_load(*args: object, **kwargs: object) -> ModelRuntime:
        raise AssertionError("Model loading must not occur for invalid input")

    monkeypatch.setattr(benchmark_generation, "load_runtime", fail_load)
    with pytest.raises(SystemExit) as error:
        benchmark_generation.main()
    assert error.value.code == 2
