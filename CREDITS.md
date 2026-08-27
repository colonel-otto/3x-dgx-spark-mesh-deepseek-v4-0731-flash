# Credits

This work builds on measurements and patches published by others. Their
methodology is reused here so results are comparable rather than merely
adjacent.

## Upstream projects

- **[MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark](https://github.com/MiaAI-Lab/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)**
  Two-node DSpark recipe. Their `scripts/bench-miaai.py` is the harness used for
  the matched 2-vs-3 node comparison in this repo, and their audit-suite design
  notes independently identified the same prompt-sensitivity effect recorded in
  [benchmark policy](docs/BENCHMARK-POLICY.md#prompt-jit-and-reference-tool-discipline).

- **[localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark)**
  Three-node TP=3 report and `benchmark_tp3.py`. The attention-group padding
  approach that makes TP=3 correct (`o_groups` 8 to 9) originates there —
  **without it, vLLM at TP=3 serves fluent nonsense rather than failing**, so
  every result in this repository depends on it.

  Their published **618 tok/s at `max_num_seqs=32`** was, for several days, the
  one figure here we could not match. We had rejected `seqs=32` as a stability
  regression. Re-tested on 2026-08-26 against a fabric that turned out to have
  been degraded when we rejected it, it reached **685.9 tok/s** — the single
  largest throughput gain in this project, and it came from taking someone
  else's number seriously enough to re-open our own conclusion.
  See [`results/20260826-seqs32-retest/`](results/20260826-seqs32-retest).

- **[NVIDIA dgx-spark-playbooks](https://github.com/NVIDIA/dgx-spark-playbooks)**
  Reference three-Spark ring topology and the switchless NCCL settings.

- **[Anemll](https://github.com/anemll)** - the `dspark-vllm-gx10` runtime image.

- **[DeepSeek-V4-Flash-0731](https://huggingface.co/deepseek-ai/DeepSeek-V4-Flash-0731)**
  (MIT) - the model measured throughout.

## Attribution in the data

Rows in [`benchmarks/measurements.csv`](benchmarks/measurements.csv) sourced
from third-party publications carry `source=external-published` and are never
mixed silently with locally measured rows.
