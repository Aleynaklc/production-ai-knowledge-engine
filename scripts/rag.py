"""Ask one question through the fully local grounded RAG pipeline."""

import argparse
import json

from backend.app.config import Settings
from backend.app.rag.factory import build_rag_runtime


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("question", help="Question to answer from the local knowledge base")
    parser.add_argument("--top-k", type=int, default=None)
    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()
    if not args.question.strip() or (args.top_k is not None and args.top_k < 1):
        parser.error("question must contain text and top-k must be positive")
    runtime = build_rag_runtime(Settings())
    try:
        answer = runtime.service.answer(args.question, args.top_k)
    finally:
        runtime.close()
    print(json.dumps(answer.model_dump(mode="json"), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
