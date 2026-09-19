"""A passing absolute score must not conceal a regression on the same questions."""

from copy import deepcopy
from typing import Any

import pytest

from scripts.compare_workspace_reports import compare_reports


def report() -> dict[str, Any]:
    return {
        "dataset_sha256": "frozen",
        "variants": {
            "current": {
                "summary": {
                    "query_count": 10,
                    "answerable_count": 8,
                    "unanswerable_count": 2,
                    "answerable_lexical_success_rate": 1.0,
                    "unanswerable_false_answers": 0,
                    "mean_evidence_span_recall": 1.0,
                    "quality_gate_passed": True,
                }
            }
        },
    }


@pytest.mark.parametrize(
    "metric,value",
    [
        ("answerable_lexical_success_rate", 0.875),
        ("unanswerable_false_answers", 1),
        ("mean_evidence_span_recall", 0.9),
    ],
)
def test_absolute_gate_cannot_hide_regression(metric: str, value: float) -> None:
    previous = report()
    current = deepcopy(previous)
    current["variants"]["current"]["summary"][metric] = value
    result = compare_reports(previous, current)
    assert result["regressions"] == [metric]
    assert not result["release_checks_passed"]


def test_matching_a_failing_baseline_does_not_pass_release_checks() -> None:
    previous = report()
    current = deepcopy(previous)
    current["variants"]["current"]["summary"]["quality_gate_passed"] = False
    result = compare_reports(previous, current)
    assert result["no_regression"] and not result["release_checks_passed"]


def test_reports_with_changed_labels_cannot_be_compared() -> None:
    previous = report()
    current = deepcopy(previous)
    current["dataset_sha256"] = "changed"
    with pytest.raises(ValueError, match="same frozen dataset"):
        compare_reports(previous, current)
    assert compare_reports(previous, previous)["release_checks_passed"]
