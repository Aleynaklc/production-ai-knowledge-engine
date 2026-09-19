"""Shared exclusion of instruction-bearing lines from factual evidence."""

import re

DIRECTIVE_PATTERN = re.compile(
    r"(?:ignore\s+(?:(?:all|the|previous|above)\s+)*(?:instructions|rules)|"
    r"answer\s+(?:every|all)\s+questions?\s+with|pretend\s+.+|"
    r"(?:system|assistant)\s*:\s*(?:ignore|override))",
    re.IGNORECASE,
)


def factual_evidence(text: str) -> str:
    """Instruction-bearing lines cannot substantiate a factual answer.

    This is a narrow defense in depth, not a complete prompt-injection detector.
    Original documents remain available in the source viewer.
    """
    return "\n".join(line for line in text.splitlines() if not DIRECTIVE_PATTERN.search(line))
