"""Generate text with the configured local Hugging Face instruct model."""

import argparse
import json

from backend.app.config import Settings
from backend.app.llm.generation import GenerationOptions, generate_text
from backend.app.llm.model import load_runtime
from backend.app.llm.tokenizer import DEFAULT_SYSTEM_PROMPT


def build_parser(settings: Settings) -> argparse.ArgumentParser:
    """Create the local-generation command-line interface."""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prompt", required=True, help="User prompt to send to the model")
    parser.add_argument("--system-prompt", default=DEFAULT_SYSTEM_PROMPT)
    parser.add_argument("--model-name", default=settings.model_name)
    parser.add_argument("--revision", default=settings.model_revision)
    parser.add_argument("--device", choices=["auto", "cpu", "mps", "cuda"], default=settings.device)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-k", type=int, default=50)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--max-new-tokens", type=int, default=settings.max_new_tokens)
    parser.add_argument("--repetition-penalty", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--greedy", action="store_true", help="Disable sampling and use greedy decoding"
    )
    parser.add_argument("--json", action="store_true", help="Print machine-readable JSON")
    return parser


def main() -> None:
    """Load the requested model, generate a response, and print measurements."""

    settings = Settings()
    args = build_parser(settings).parse_args()
    options = GenerationOptions(
        temperature=args.temperature,
        top_k=args.top_k,
        top_p=args.top_p,
        max_new_tokens=args.max_new_tokens,
        repetition_penalty=args.repetition_penalty,
        seed=args.seed,
        do_sample=not args.greedy,
    )
    if not args.json:
        print(f"Loading {args.model_name} on {args.device}...", flush=True)
    runtime = load_runtime(args.model_name, args.revision, args.device)
    result = generate_text(runtime, args.prompt, options, args.system_prompt)
    payload = {"model": runtime.metadata(), "generation": result.model_dump()}

    if args.json:
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return

    model_metadata = runtime.metadata()
    print(f"Model:              {model_metadata['model_name']}")
    print(f"Revision:           {model_metadata['resolved_revision']}")
    print(f"Device / dtype:     {model_metadata['device']} / {model_metadata['dtype']}")
    print(f"Parameters:         {model_metadata['parameter_count']:,}")
    print(f"Model load time:    {model_metadata['load_seconds']:.3f} s")
    print(f"Input tokens:       {result.input_tokens}")
    print(f"Output tokens:      {result.output_tokens}")
    print(f"Generation time:    {result.generation_seconds:.3f} s")
    print(f"Output tokens/sec:  {result.tokens_per_second:.3f}")
    print("\nAnswer:\n")
    print(result.text)


if __name__ == "__main__":
    main()
