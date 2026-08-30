"""Unit tests for generation configuration."""

import pytest
from pydantic import ValidationError

from backend.app.llm.generation import GenerationOptions, build_generation_kwargs


def test_sampling_kwargs_include_probability_controls() -> None:
    """Sampling should forward temperature, top-k, and top-p explicitly."""

    options = GenerationOptions(temperature=1.2, top_k=25, top_p=0.8)

    kwargs = build_generation_kwargs(options, pad_token_id=1, eos_token_id=2)

    assert kwargs["do_sample"] is True
    assert kwargs["temperature"] == 1.2
    assert kwargs["top_k"] == 25
    assert kwargs["top_p"] == 0.8
    assert kwargs["use_cache"] is True


def test_greedy_kwargs_omit_inactive_sampling_controls() -> None:
    """Greedy decoding should not pass ignored sampling arguments to Transformers."""

    options = GenerationOptions(do_sample=False)

    kwargs = build_generation_kwargs(options, pad_token_id=1, eos_token_id=2)

    assert kwargs["do_sample"] is False
    assert "temperature" not in kwargs
    assert "top_k" not in kwargs
    assert "top_p" not in kwargs


@pytest.mark.parametrize(
    ("field", "value"),
    [("temperature", 0), ("top_p", 1.1), ("max_new_tokens", 0), ("repetition_penalty", 0)],
)
def test_generation_options_reject_invalid_values(field: str, value: float) -> None:
    """Invalid decoding values should fail before expensive model inference."""

    with pytest.raises(ValidationError):
        GenerationOptions.model_validate({field: value})
