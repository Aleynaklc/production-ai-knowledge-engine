"""Execution test for the Stage 2 tokenization notebook."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path("notebooks/02_tokenization_inspection.ipynb")


def test_tokenization_notebook_executes_without_errors() -> None:
    """Tokenizer loading, comparisons, and assertions should all succeed."""

    # nbformat does not currently type its public read helper.
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)  # type: ignore[no-untyped-call]
    client = NotebookClient(
        notebook,
        timeout=180,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    executed_notebook = client.execute()

    assert all(
        output.get("output_type") != "error"
        for cell in executed_notebook.cells
        for output in cell.get("outputs", [])
    )
