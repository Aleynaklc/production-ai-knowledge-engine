"""Tests for objective Stage 4 experiment checks."""

from scripts.compare_generation import follows_requested_format


def test_requested_format_requires_three_em_dash_lines() -> None:
    """Exactly three populated name/explanation lines should pass."""

    text = "Alpha — first name\nBeta — second name\nGamma — third name"

    assert follows_requested_format(text) is True


def test_requested_format_rejects_single_paragraph() -> None:
    """Mentioning an em dash is insufficient without three separate lines."""

    assert follows_requested_format("Alpha — one long paragraph") is False
