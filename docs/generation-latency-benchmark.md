# Generation Latency Benchmark — Stage 26

The benchmark compares short, medium, and long prompts against 32- and 64-token
output caps on the configured, revision-pinned causal language model. Exact prompts,
SHA-256 hashes, model revision, device, dtype, library versions, CPU thread counts,
seeds, outputs, and execution order are recorded in the JSON report.

Run the complete local matrix:

```bash
uv run python -m scripts.benchmark_generation \
  --prompt-lengths short medium long \
  --output-lengths 32 64 \
  --repeats 3 --warmup-runs 1 --seed 42
```

Outputs:

- `evaluation/reports/generation_latency.json`: every measured trial and aggregate.
- `evaluation/reports/generation_latency.md`: measured comparison table and environment.

For a shorter smoke measurement, use `--prompt-lengths short long --output-lengths
16 32 --repeats 2`. More repetitions, such as `--repeats 20`, are needed for a more
stable p95. `--device cpu|mps|cuda|auto`, `--model-name`, and `--revision` select the
runtime; revisions should be immutable model commits for reproducible comparisons.
Output destinations can be changed with `--output` and `--markdown-output`.

## Measurement definitions

The model loads once; its separate wall-time measurement includes tokenizer loading,
model loading, device transfers, and a final accelerator synchronization. Downloads
and cache access can affect this number, so it is not inherently a cold-start or a
cached-load measurement. It never contributes to generation latency summaries.

Each profile receives its own discarded warmup calls. Measured repetitions use a
seeded shuffle of the profiles to reduce fixed-order effects. Runs are sequential
with a batch size of one. Greedy decoding is the default; `--sample` enables seeded
sampling using the recorded `GenerationOptions`. A measured trial's seed is the base
seed plus its zero-based repetition index, modulo 2³². Identical seeds do not ensure
bitwise deterministic output across hardware and library versions.

- **Generation p50/p95:** synchronized `model.generate` duration, including prefill
  and autoregressive decoding. Tokenization and output text decoding are excluded.
- **Request p50/p95:** wall time around `generate_text`, including prompt processing,
  device transfers, generation, and decoding the output text.
- **Output tokens/sec:** sum of actual generated tokens divided by summed generation
  time for that profile. This is not an average of per-request ratios.
- **Actual output lengths and cap hit rate:** the cap is a maximum, not a forced
  output length. EOS can stop a response early; generated special tokens count toward
  the measured token total even when they are absent from decoded text.
- **Percentiles:** NumPy's linear interpolation, evaluated per profile using measured
  runs only. A few repetitions are smoke evidence, not a production tail-latency claim.
- **TTFT:** explicitly unavailable (`null`). The existing non-streaming generation
  implementation does not timestamp the first emitted token. Neither full generation
  latency nor total latency divided by token count is presented as TTFT.

The default matrix contains six profiles, six discarded warmups, and eighteen
measured calls. `--warmup-runs 0` is supported for investigation, but then the first
measured call may include initialization effects. Do not compare runs performed
alongside other model benchmarks or heavy workloads as if they were isolated runs.

## Validation

```bash
uv run pytest tests/evaluation/test_generation_latency.py
```

Unit tests use fake generators to verify discarded warmups, deterministic trial
order and seeds, real-token throughput arithmetic, percentile aggregation, early EOS,
invalid input rejection before model loading, and report serialization. Their
synthetic timings are never written as model benchmark evidence.
