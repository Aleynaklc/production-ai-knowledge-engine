# Local Hugging Face Inference Evidence

This report records the Stage 3 acceptance run. Values are measurements from this
machine and should not be generalized to other hardware.

## Configuration

| Field | Value |
|---|---|
| Model | `Qwen/Qwen2.5-0.5B-Instruct` |
| Revision | `c89bee90d9f811437d9735454613c35b4a3c4dc8` |
| Parameters | 494,032,768 |
| Device | MPS |
| Dtype | FP16 |
| Python | 3.12.13 |
| PyTorch | 2.13.0 |
| Transformers | 5.16.1 |

## Command

```bash
uv run python -m scripts.generate \
  --prompt "Explain vector search in exactly one concise sentence." \
  --greedy \
  --max-new-tokens 64
```

## Measured result

| Metric | Value |
|---|---:|
| Cached model load | 2.353 s |
| Input tokens | 32 |
| Output tokens | 28 |
| Generation time | 1.480 s |
| Output tokens/s | 18.916 |

Generated answer:

> Vector search involves using vectors to find specific elements within a dataset,
> often in the context of computer science, mathematics, or data analysis.

The first run reported 146.744 seconds of model load time because it included the
initial model download; the cached run above is the useful local load measurement.
Throughput is defined as `output tokens / model.generate elapsed seconds`. It excludes
model loading and is not Time To First Token, because this stage uses non-streaming
generation.

The response demonstrates functioning local autoregressive inference, not answer
quality evaluation. Grounding and domain-specific retrieval are introduced in later
stages.
