# Generation Latency Benchmark — Stage 26

Measured at: `2026-09-10T09:37:32.208589+00:00`

Model: `Qwen/Qwen2.5-0.5B-Instruct`; resolved revision: `c89bee90d9f811437d9735454613c35b4a3c4dc8`; requested revision: `c89bee90d9f811437d9735454613c35b4a3c4dc8`.
Device/dtype: `cpu` / `float32`. Model load: **2.551 s**, reported separately.
3 measured repetitions per profile; 6 total warmup runs discarded. Base seed: `42`; sampling: `False`.

| Profile | Input tokens | Actual output tokens | Generation p50 (s) | Generation p95 (s) | Request p50 (s) | Request p95 (s) | Output tok/s | Cap hit rate |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| short_output_32 | 82–82 | 32–32 | 2.387 | 3.261 | 2.387 | 3.269 | 11.883 | 100.0% |
| short_output_64 | 82–82 | 64–64 | 4.767 | 5.121 | 4.768 | 5.124 | 13.284 | 100.0% |
| medium_output_32 | 341–341 | 32–32 | 2.745 | 6.879 | 2.749 | 6.884 | 7.486 | 100.0% |
| medium_output_64 | 341–341 | 64–64 | 5.071 | 5.476 | 5.073 | 5.480 | 12.359 | 100.0% |
| long_output_32 | 948–948 | 32–32 | 3.339 | 3.517 | 3.344 | 3.524 | 9.433 | 100.0% |
| long_output_64 | 948–948 | 64–64 | 5.710 | 5.878 | 5.715 | 5.882 | 11.130 | 100.0% |

TTFT: **not measured**. The existing non-streaming generate_text call has no first-token timestamp. Full generation latency is not a TTFT estimate.

- Generation timing brackets synchronized model.generate, including prefill and autoregressive decoding; it excludes prompt tokenization and output text decoding.
- Request timing brackets generate_text, including tokenization, device transfer, generation, and output text decoding. Model loading and warmups are excluded.
- Output tokens/sec is sum(actual output tokens) / sum(generation seconds) per profile. The output cap is a maximum; EOS may end generation earlier, and token counts include generated special tokens.
- Percentiles use NumPy linear interpolation. Small repeat counts provide smoke measurements, not reliable production tail-latency estimates.
- Trials are single-request, batch size one, with seeded shuffled profile order in each repetition. Seed equals base seed plus repetition modulo 2**32. Greedy is the default; fixed seeds do not guarantee bitwise determinism across hardware.

Model-load boundary: CLI wall time for tokenizer/model loading and final device synchronization; may include downloads and cache access.

Environment:

- platform: `macOS-26.6.2-arm64-arm-64bit`
- machine: `arm64`
- processor_or_accelerator: `arm`
- python: `3.12.13`
- torch: `2.13.0`
- transformers: `5.16.1`
- torch_num_threads: `4`
- torch_num_interop_threads: `8`

The companion JSON contains the exact prompts, their SHA-256 hashes, every measured output, per-trial options/seeds, execution order, and model metadata.
