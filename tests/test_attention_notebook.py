"""Execution test for the Stage 1 attention notebook."""

from pathlib import Path

import nbformat
from nbclient import NotebookClient

NOTEBOOK_PATH = Path("notebooks/01_attention_from_scratch.ipynb")


def test_attention_notebook_executes_without_errors() -> None:
    """Every attention experiment and assertion should execute successfully."""

    # nbformat does not currently type its public read helper.
    notebook = nbformat.read(NOTEBOOK_PATH, as_version=4)  # type: ignore[no-untyped-call]
    client = NotebookClient(
        notebook,
        timeout=120,
        kernel_name="python3",
        resources={"metadata": {"path": str(NOTEBOOK_PATH.parent)}},
    )

    executed_notebook = client.execute()

    assert all(
        output.get("output_type") != "error"
        for cell in executed_notebook.cells
        for output in cell.get("outputs", [])
    )
