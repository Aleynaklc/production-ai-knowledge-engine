"""UTF-8 Markdown and plain-text loading."""

from pathlib import Path


class DocumentLoadError(RuntimeError):
    """Raised when a supported text document cannot be read."""


def load_text_file(path: Path) -> str:
    """Load a UTF-8 or UTF-8-with-BOM text document."""

    try:
        return path.read_text(encoding="utf-8-sig")
    except (OSError, UnicodeDecodeError) as error:
        raise DocumentLoadError(f"Could not read {path}: {error}") from error
