"""Conservative text normalization that preserves document meaning."""

import re
import unicodedata


def clean_text(text: str) -> str:
    """Normalize Unicode, line endings, trailing spaces, and excessive blank lines."""

    normalized = unicodedata.normalize("NFC", text).replace("\r\n", "\n").replace("\r", "\n")
    without_trailing_space = "\n".join(line.rstrip() for line in normalized.splitlines())
    compact = re.sub(r"\n{3,}", "\n\n", without_trailing_space)
    return compact.strip()
