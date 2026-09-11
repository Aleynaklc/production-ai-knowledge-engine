"""Benchmark generation latency across prompt lengths and output caps (Stage 26)."""

import argparse
from pathlib import Path
from time import perf_counter

from pydantic import ValidationError

from backend.app.config import Settings
from backend.app.evaluation.generation_latency import (
    GenerationBenchmarkConfig,
    build_generation_profiles,
    render_generation_benchmark,
    run_generation_benchmark,
    synchronize_runtime,
)
from backend.app.llm.model import load_runtime
from backend.app.llm.tokenizer import DEFAULT_SYSTEM_PROMPT


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """Create a CLI with validated inputs before any expensive model loading."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model-name", default=settings.model_name)
    parser.add_argument("--revision", default=settings.model_revision)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=settings.device)
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--sample", action="store_true", help="Use seeded sampling instead of greedy"
    )
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument(
        "--prompt-lengths",
        nargs="+",
        choices=["short", "medium", "long"],
        default=["short", "medium", "long"],
    )
    parser.add_argument("--output-lengths", nargs="+", type=int, default=[32, 64])
    parser.add_argument(
        "--output", type=Path, default=Path("evaluation/reports/generation_latency.json")
    )
    parser.add_argument(
        "--markdown-output",
        type=Path,
        default=Path("evaluation/reports/generation_latency.md"),
    )
    return parser


def main() -> None:
    """Load once, measure real model calls, then persist JSON and Markdown evidence."""

    parser = build_parser(Settings())
    args = parser.parse_args()
    try:
        config = GenerationBenchmarkConfig(
            repeats=args.repeats,
            warmup_runs=args.warmup_runs,
            seed=args.seed,
            do_sample=args.sample,
            system_prompt=args.system_prompt,
        )
        profiles = build_generation_profiles(args.output_lengths, args.prompt_lengths)
        if args.output.resolve() == args.markdown_output.resolve():
            raise ValueError("JSON and Markdown output paths must be different")
    except (ValidationError, ValueError) as error:
        parser.error(str(error))

    print(f"Loading {args.model_name} on {args.device}...", flush=True)
    started_at = perf_counter()
    runtime = load_runtime(args.model_name, args.revision, args.device)
    synchronize_runtime(runtime)
    model_load_seconds = perf_counter() - started_at
    print(
        f"Running {len(profiles)} profiles × {config.repeats} measured repetitions "
        f"plus {config.warmup_runs} warmups per profile...",
        flush=True,
    )
    report = run_generation_benchmark(
        runtime, profiles, config, model_load_seconds=model_load_seconds
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.markdown_output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(report.model_dump_json(indent=2) + "\n", encoding="utf-8")
    args.markdown_output.write_text(render_generation_benchmark(report), encoding="utf-8")
    print(f"Wrote {args.output}")
    print(f"Wrote {args.markdown_output}")


if __name__ == "__main__":
    main()
