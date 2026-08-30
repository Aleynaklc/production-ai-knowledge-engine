"""Configurable Hugging Face causal language-model loading."""

from collections.abc import Iterator
from dataclasses import dataclass
from time import perf_counter
from typing import Literal, Protocol, cast

import torch
from transformers import AutoModelForCausalLM, PreTrainedTokenizerBase

from backend.app.llm.tokenizer import load_tokenizer

DeviceRequest = Literal["auto", "cpu", "mps", "cuda"]


class DeviceUnavailableError(RuntimeError):
    """Raised when a requested accelerator is not available."""


class CausalLanguageModel(Protocol):
    """Small typed boundary around the dynamic Transformers model API."""

    config: object

    def to(self, device: torch.device) -> object:
        """Move parameters to a device."""

    def eval(self) -> object:
        """Enable inference behavior."""

    def parameters(self) -> Iterator[torch.nn.Parameter]:
        """Iterate over model parameters."""

    def generate(self, **kwargs: object) -> torch.Tensor:
        """Generate token sequences."""


@dataclass(frozen=True, slots=True)
class ModelRuntime:
    """Loaded model, tokenizer, and reproducibility metadata."""

    model: CausalLanguageModel
    tokenizer: PreTrainedTokenizerBase
    model_name: str
    requested_revision: str
    resolved_revision: str | None
    device: torch.device
    dtype: torch.dtype
    load_seconds: float
    parameter_count: int
    parameter_bytes: int

    def metadata(self) -> dict[str, str | int | float | None]:
        """Return JSON-compatible model metadata."""

        return {
            "model_name": self.model_name,
            "requested_revision": self.requested_revision,
            "resolved_revision": self.resolved_revision,
            "device": self.device.type,
            "dtype": str(self.dtype).removeprefix("torch."),
            "load_seconds": round(self.load_seconds, 6),
            "parameter_count": self.parameter_count,
            "parameter_bytes": self.parameter_bytes,
        }


def resolve_device(requested: DeviceRequest) -> torch.device:
    """Resolve an explicit device or choose the best available accelerator."""

    if requested == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise DeviceUnavailableError("CUDA was requested but is not available")
    if requested == "mps" and not torch.backends.mps.is_available():
        raise DeviceUnavailableError("MPS was requested but is not available")
    return torch.device(requested)


def select_dtype(device: torch.device) -> torch.dtype:
    """Choose a conservative inference dtype for the resolved device."""

    if device.type == "cuda":
        return torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16
    if device.type == "mps":
        return torch.float16
    return torch.float32


def load_runtime(
    model_name: str,
    revision: str,
    requested_device: DeviceRequest = "auto",
) -> ModelRuntime:
    """Load a tokenizer and causal LM without relying on the pipeline helper."""

    started_at = perf_counter()
    device = resolve_device(requested_device)
    dtype = select_dtype(device)
    tokenizer = load_tokenizer(model_name, revision)
    model = cast(
        CausalLanguageModel,
        AutoModelForCausalLM.from_pretrained(
            model_name,
            revision=revision,
            dtype=dtype,
            low_cpu_mem_usage=True,
        ),
    )
    model.to(device)
    model.eval()

    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    parameter_bytes = sum(
        parameter.numel() * parameter.element_size() for parameter in model.parameters()
    )
    resolved_revision = cast(str | None, getattr(model.config, "_commit_hash", None))
    return ModelRuntime(
        model=model,
        tokenizer=tokenizer,
        model_name=model_name,
        requested_revision=revision,
        resolved_revision=resolved_revision,
        device=device,
        dtype=dtype,
        load_seconds=perf_counter() - started_at,
        parameter_count=parameter_count,
        parameter_bytes=parameter_bytes,
    )
