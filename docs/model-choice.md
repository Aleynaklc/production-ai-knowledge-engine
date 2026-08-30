# Local Model Choice

## Context

Stages 3 and 4 require a small public instruct model that works through Hugging Face
`AutoTokenizer` and `AutoModelForCausalLM`, can run without a hosted API, and is
practical for repeated generation experiments on a local arm64 Mac.

## Decision

Use `Qwen/Qwen2.5-0.5B-Instruct`, pinned to revision
`c89bee90d9f811437d9735454613c35b4a3c4dc8`.

The official model card describes it as a 0.49B-parameter, instruction-tuned causal
language model with an Apache-2.0 license and a 32,768-token configured context. It is
small enough for CPU development, already uses the tokenizer inspected in Stage 2,
and requires no model-specific remote Python code.

Source: [official Hugging Face model card](https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct).

## Device behavior

The application resolves `auto` in this order:

1. CUDA
2. Apple MPS
3. CPU

PyTorch documents `torch.backends.mps.is_available()` as the runtime availability
check for Metal acceleration. A restricted preflight process could not access MPS,
while the approved inference process resolved MPS successfully. The recorded Stage 4
measurements therefore use MPS/FP16. Device availability is checked inside each
process; `auto` falls back to CPU/FP32 when no accelerator is accessible.

Source: [PyTorch MPS backend documentation](https://docs.pytorch.org/docs/stable/notes/mps.html).

## Alternatives and trade-offs

- A larger Qwen, Gemma, Mistral, Llama, or Phi model may improve output quality, but
  increases download size, memory pressure, and experiment latency.
- A smaller model can shorten iteration time, but may weaken instruction following.
- A hosted model API would simplify local hardware requirements, but would not prove
  local Hugging Face model loading or expose local inference measurements.
- CPU/FP32 is broadly compatible but uses more parameter memory and is generally
  slower than a supported reduced-precision accelerator path. Actual performance is
  measured rather than assumed.

## Safetensors awareness

The selected repository distributes weights in the Safetensors format. Model files
remain external cache artifacts and are excluded from Git; the repository records the
model identifier and immutable revision instead of committing weights.
