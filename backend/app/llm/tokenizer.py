"""Tokenizer loading and prompt construction for local instruct models."""

from collections.abc import Mapping
from typing import cast

import torch
from transformers import AutoTokenizer, PreTrainedTokenizerBase

DEFAULT_SYSTEM_PROMPT = "You are a concise and helpful AI assistant."


def load_tokenizer(model_name: str, revision: str) -> PreTrainedTokenizerBase:
    """Load the fast tokenizer pinned to the requested model revision."""

    tokenizer = cast(
        PreTrainedTokenizerBase,
        AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            use_fast=True,
        ),
    )
    if tokenizer.pad_token_id is None:
        if tokenizer.eos_token is None:
            raise ValueError("Tokenizer has neither a padding token nor an end-of-sequence token")
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def build_chat_inputs(
    tokenizer: PreTrainedTokenizerBase,
    prompt: str,
    system_prompt: str,
    device: torch.device,
) -> dict[str, torch.Tensor]:
    """Render a model-native chat prompt and move its tensors to the target device."""

    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt},
    ]
    if tokenizer.chat_template is None:
        rendered = tokenizer(
            f"System: {system_prompt}\nUser: {prompt}\nAssistant:",
            return_tensors="pt",
        )
    else:
        rendered = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        )

    tensor_mapping = cast(Mapping[str, torch.Tensor], rendered)
    inputs = {name: tensor.to(device) for name, tensor in tensor_mapping.items()}
    if "input_ids" not in inputs:
        raise ValueError("Tokenizer output did not contain input_ids")
    return inputs
