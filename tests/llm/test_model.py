"""Unit tests for model device and dtype selection."""

import torch

from backend.app.llm.model import resolve_device, select_dtype


def test_cpu_device_uses_float32() -> None:
    """The conservative CPU path should avoid unsupported reduced precision."""

    device = resolve_device("cpu")

    assert device.type == "cpu"
    assert select_dtype(device) is torch.float32


def test_auto_device_resolves_to_an_available_backend() -> None:
    """Automatic selection should always produce a usable local backend."""

    assert resolve_device("auto").type in {"cpu", "mps", "cuda"}
