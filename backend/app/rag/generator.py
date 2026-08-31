"""Language-model adapter for deterministic grounded answer generation."""

from typing import Protocol

from backend.app.llm.generation import GenerationOptions, GenerationResult, generate_text
from backend.app.llm.model import ModelRuntime


class GroundedGenerator(Protocol):
    """Generate one answer from a grounded prompt contract."""

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        """Return generated text and token/latency measurements."""


class LocalGroundedGenerator:
    """Use the project's pinned local causal model for grounded generation."""

    def __init__(self, runtime: ModelRuntime, options: GenerationOptions) -> None:
        self.runtime = runtime
        self.options = options

    def generate(self, prompt: str, system_prompt: str) -> GenerationResult:
        """Generate a deterministic answer with the shared inference implementation."""

        return generate_text(
            self.runtime,
            prompt,
            self.options,
            system_prompt=system_prompt,
        )
