"""Compare frozen-dataset model runs without hiding regressions behind an absolute gate."""

import argparse
import json
from pathlib import Path
from typing import Any


def compare_reports(previous: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    if not previous.get("dataset_sha256") or previous["dataset_sha256"] != current.get(
        "dataset_sha256"
    ):
        raise ValueError("Reports must use the same frozen dataset")
    before = previous["variants"]["current"]["summary"]
    after = current["variants"]["current"]["summary"]
    for key in ("query_count", "answerable_count", "unanswerable_count"):
        if before[key] != after[key]:
            raise ValueError("Report query counts differ")
    regressions = []
    deltas: dict[str, float | None] = {}
    for key, higher_is_better in (
        ("answerable_lexical_success_rate", True),
        ("unanswerable_false_answers", False),
        ("mean_evidence_span_recall", True),
    ):
        old, new = before.get(key), after.get(key)
        if old is None or new is None:
            deltas[key] = None
            continue
        delta = float(new) - float(old)
        deltas[key] = delta
        if (higher_is_better and delta < -1e-12) or (not higher_is_better and delta > 1e-12):
            regressions.append(key)
    return {
        "dataset_sha256": current["dataset_sha256"],
        "no_regression": not regressions,
        "regressions": regressions,
        "deltas": deltas,
        "current_quality_gate_passed": after["quality_gate_passed"],
        "release_checks_passed": not regressions and after["quality_gate_passed"],
        "note": "Lexical/source regression checks are not semantic correctness guarantees. Latency is excluded from this uncontrolled comparison.",
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("previous", type=Path)
    parser.add_argument("current", type=Path)
    args = parser.parse_args()
    try:
        result = compare_reports(
            json.loads(args.previous.read_text()), json.loads(args.current.read_text())
        )
    except (ValueError, KeyError, OSError) as error:
        parser.error(str(error))
    print(json.dumps(result, indent=2))
    return int(not result["release_checks_passed"])


if __name__ == "__main__":
    raise SystemExit(main())
