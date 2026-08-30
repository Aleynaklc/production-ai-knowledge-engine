"""Run reproducible sampling experiments and write Stage 4 reports."""

import argparse
import json
import platform
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import torch
import transformers

from backend.app.config import Settings
from backend.app.llm.generation import GenerationOptions, generate_text
from backend.app.llm.model import ModelRuntime, load_runtime

DEFAULT_PROMPT = (
    "Name an internal AI knowledge engine. Return exactly three lines in the format "
    "'Name — explanation', with each explanation limited to eight words."
)
SYSTEM_PROMPT = (
    "You are a concise product-naming assistant. Follow the requested format exactly "
    "and add no introduction."
)
JSON_REPORT = Path("evaluation/reports/generation_sampling.json")
MARKDOWN_REPORT = Path("docs/generation-sampling.md")


def build_profiles(max_new_tokens: int) -> list[tuple[str, GenerationOptions]]:
    """Define temperature, nucleus-sampling, and deterministic baselines."""

    def options(
        *,
        temperature: float = 0.7,
        top_p: float = 0.9,
        do_sample: bool = True,
    ) -> GenerationOptions:
        return GenerationOptions(
            max_new_tokens=max_new_tokens,
            top_k=50,
            seed=42,
            temperature=temperature,
            top_p=top_p,
            do_sample=do_sample,
        )

    return [
        ("greedy", options(do_sample=False)),
        ("temperature_0.1", options(temperature=0.1)),
        ("temperature_0.7", options(temperature=0.7)),
        ("temperature_0.7_repeat", options(temperature=0.7)),
        ("temperature_1.2", options(temperature=1.2)),
        ("top_p_0.5", options(top_p=0.5)),
        ("top_p_0.95", options(top_p=0.95)),
    ]


def follows_requested_format(text: str) -> bool:
    """Check the experiment's exact three-line, em-dash-separated format."""

    nonempty_lines = [line.strip() for line in text.splitlines() if line.strip()]
    return len(nonempty_lines) == 3 and all(" — " in line for line in nonempty_lines)


def run_experiments(
    runtime: ModelRuntime,
    prompt: str,
    max_new_tokens: int,
) -> list[dict[str, Any]]:
    """Run every profile against one fixed prompt and collect real measurements."""

    records: list[dict[str, Any]] = []
    for profile_name, options in build_profiles(max_new_tokens):
        print(f"Running {profile_name}...", flush=True)
        result = generate_text(runtime, prompt, options, SYSTEM_PROMPT)
        records.append(
            {
                "profile": profile_name,
                **result.model_dump(),
                "format_compliant": follows_requested_format(result.text),
            }
        )
    return records


def render_markdown(
    runtime: ModelRuntime,
    prompt: str,
    records: list[dict[str, Any]],
    created_at: str,
) -> str:
    """Render a human-readable report from the JSON-compatible records."""

    metadata = runtime.metadata()
    parameter_bytes = metadata["parameter_bytes"]
    if not isinstance(parameter_bytes, int):
        raise TypeError("Model metadata parameter_bytes must be an integer")
    repeated_is_identical = records[2]["text"] == records[3]["text"]
    compliant_count = sum(bool(record["format_compliant"]) for record in records)
    lines = [
        "# Generation Sampling Comparison",
        "",
        f"Generated at: `{created_at}`",
        "",
        "## Reproducibility",
        "",
        f"- Model: `{metadata['model_name']}`",
        f"- Revision: `{metadata['resolved_revision']}`",
        f"- Device / dtype: `{metadata['device']}` / `{metadata['dtype']}`",
        f"- Parameters: `{metadata['parameter_count']:,}`",
        f"- Parameter memory: `{parameter_bytes / (1024**3):.3f} GiB`",
        f"- Python: `{platform.python_version()}`",
        f"- PyTorch: `{torch.__version__}`",
        f"- Transformers: `{transformers.__version__}`",
        f"- Platform: `{platform.platform()}`",
        "",
        "## Fixed prompt",
        "",
        f"> {prompt}",
        "",
        "## Results",
        "",
        "| Profile | Strategy | Temp | Top-p | Format | Output tokens | Seconds | tok/s |",
        "|---|---|---:|---:|---|---:|---:|---:|",
    ]
    for record in records:
        options = record["options"]
        strategy = "sample" if options["do_sample"] else "greedy"
        temperature = f"{options['temperature']:.1f}" if options["do_sample"] else "—"
        top_p = f"{options['top_p']:.2f}" if options["do_sample"] else "—"
        lines.append(
            f"| {record['profile']} | {strategy} | {temperature} | {top_p} | "
            f"{'yes' if record['format_compliant'] else 'no'} | "
            f"{record['output_tokens']} | {record['generation_seconds']:.3f} | "
            f"{record['tokens_per_second']:.3f} |"
        )
    lines.extend(
        [
            "",
            "The two `temperature_0.7` runs use the same seed and are "
            f"**{'identical' if repeated_is_identical else 'different'}** in this environment.",
            f"Only **{compliant_count}/{len(records)}** profiles followed the requested exact "
            "three-line format. Sampling controls alter token selection; they do not guarantee "
            "instruction following, especially for a small model.",
            "Greedy decoding ignores temperature and top-p because it always selects the "
            "highest-scoring next token. Sampling changes the logits distribution before a "
            "seeded random draw; higher temperature usually flattens it, while lower top-p "
            "restricts the candidate probability mass. These settings affect diversity, not "
            "guaranteed factual quality.",
            "",
            "Throughput is defined as `output tokens / model.generate elapsed seconds`. Model "
            "load time is excluded. These are local single-run measurements, not production "
            "latency claims.",
            "",
            "## Full outputs",
            "",
        ]
    )
    for record in records:
        lines.extend([f"### {record['profile']}", "", str(record["text"]), ""])
    return "\n".join(lines)


def write_reports(
    runtime: ModelRuntime,
    prompt: str,
    records: list[dict[str, Any]],
) -> None:
    """Write machine-readable and human-readable experiment evidence."""

    created_at = datetime.now(UTC).isoformat()
    payload = {
        "created_at": created_at,
        "model": runtime.metadata(),
        "environment": {
            "platform": platform.platform(),
            "python": platform.python_version(),
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
        "prompt": prompt,
        "system_prompt": SYSTEM_PROMPT,
        "results": records,
    }
    JSON_REPORT.parent.mkdir(parents=True, exist_ok=True)
    MARKDOWN_REPORT.parent.mkdir(parents=True, exist_ok=True)
    JSON_REPORT.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    MARKDOWN_REPORT.write_text(
        render_markdown(runtime, prompt, records, created_at) + "\n",
        encoding="utf-8",
    )


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """Create the experiment command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", default=DEFAULT_PROMPT)
    parser.add_argument("--model-name", default=settings.model_name)
    parser.add_argument("--revision", default=settings.model_revision)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=settings.device)
    parser.add_argument("--max-new-tokens", type=int, default=56)
    return parser


def main() -> None:
    """Load the model once, run all profiles, and persist reports."""

    settings = Settings()
    args = build_parser(settings).parse_args()
    print(f"Loading {args.model_name} on {args.device}...", flush=True)
    runtime = load_runtime(args.model_name, args.revision, args.device)
    records = run_experiments(runtime, args.prompt, args.max_new_tokens)
    write_reports(runtime, args.prompt, records)
    print(f"Wrote {JSON_REPORT}")
    print(f"Wrote {MARKDOWN_REPORT}")


if __name__ == "__main__":
    main()
