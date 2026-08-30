# Generation Sampling Comparison

Generated at: `2026-08-27T14:29:18.687602+00:00`

## Reproducibility

- Model: `Qwen/Qwen2.5-0.5B-Instruct`
- Revision: `c89bee90d9f811437d9735454613c35b4a3c4dc8`
- Device / dtype: `mps` / `float16`
- Parameters: `494,032,768`
- Parameter memory: `0.920 GiB`
- Python: `3.12.13`
- PyTorch: `2.13.0`
- Transformers: `5.16.1`
- Platform: `macOS-26.5.2-arm64-arm-64bit`

## Fixed prompt

> Name an internal AI knowledge engine. Return exactly three lines in the format 'Name — explanation', with each explanation limited to eight words.

## Results

| Profile | Strategy | Temp | Top-p | Format | Output tokens | Seconds | tok/s |
|---|---|---:|---:|---|---:|---:|---:|
| greedy | greedy | — | — | no | 22 | 1.285 | 17.116 |
| temperature_0.1 | sample | 0.1 | 0.90 | no | 26 | 1.056 | 24.613 |
| temperature_0.7 | sample | 0.7 | 0.90 | no | 44 | 1.749 | 25.164 |
| temperature_0.7_repeat | sample | 0.7 | 0.90 | no | 44 | 1.768 | 24.889 |
| temperature_1.2 | sample | 1.2 | 0.90 | no | 16 | 0.642 | 24.935 |
| top_p_0.5 | sample | 0.7 | 0.50 | yes | 38 | 1.497 | 25.392 |
| top_p_0.95 | sample | 0.7 | 0.95 | no | 44 | 1.749 | 25.150 |

The two `temperature_0.7` runs use the same seed and are **identical** in this environment.
Only **1/7** profiles followed the requested exact three-line format. Sampling controls alter token selection; they do not guarantee instruction following, especially for a small model.
Greedy decoding ignores temperature and top-p because it always selects the highest-scoring next token. Sampling changes the logits distribution before a seeded random draw; higher temperature usually flattens it, while lower top-p restricts the candidate probability mass. These settings affect diversity, not guaranteed factual quality.

Throughput is defined as `output tokens / model.generate elapsed seconds`. Model load time is excluded. These are local single-run measurements, not production latency claims.

## Full outputs

### greedy

Assistant — A tool that provides structured information and insights from internal sources, enhancing decision-making and knowledge management.

### temperature_0.1

Assistant — A tool that provides structured, contextually appropriate information to users, often NodeList, providing insights and answers to questions.

### temperature_0.7

TensorFlow — It's an open-source AI framework that allows developers to build, NodeList — A tool for managing and analyzing large datasets, including text, image, and video, for machine learning and natural language processing.

### temperature_0.7_repeat

TensorFlow — It's an open-source AI framework that allows developers to build, NodeList — A tool for managing and analyzing large datasets, including text, image, and video, for machine learning and natural language processing.

### temperature_1.2

Ezviz — accessible knowledge application, user-guided information retrieval platform.

### top_p_0.5

TensorFlow — a powerful, open-source AI framework.  
Microsoft Azure Cognitive Services NodeList — a comprehensive suite of AI services.  
IBM Watson — a deep learning platform for enterprise AI.

### top_p_0.95

TensorFlow — It's an open-source AI framework that allows developers to build, NodeList — A tool for managing and analyzing large datasets, including entities like nodes in a graph, for machine learning and data science tasks.

