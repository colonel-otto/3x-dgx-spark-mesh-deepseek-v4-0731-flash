# Industry-Standard Benchmarks: GuideLLM

To complement custom depth and concurrency sweeps with open, vendor-neutral tooling, this repository adopts **[GuideLLM](https://github.com/vllm-project/guidellm)** (the standard serving benchmark maintained within the vLLM and Neural Magic ecosystem).

---

## Why GuideLLM

Rather than measuring only synthetic steady-state throughput, GuideLLM captures the full multi-dimensional latency and concurrency profile against the live OpenAI-compatible endpoint (`/v1/chat/completions`):

1. **Token-Level Latency**:
   - **TTFT** (Time to First Token) distributions: p50, p90, p95, p99.
   - **ITL** (Inter-Token Latency): measures streaming jitter and smoothness between chunks.
   - **TPOT** (Time per Output Token): verification of generation compute efficiency.
2. **Workload Strategies**:
   - **Concurrent Streams**: Sweep concurrent clients ($cc \in [1..32]$) to plot SLA saturation curves.
   - **Poisson Arrival Rates**: Simulate real-world interactive user query arrival patterns.
   - **Multi-Turn & Streaming**: Direct evaluation of streaming chunks and token-by-token emission.

---

## Quickstart & Execution

GuideLLM is installed in a dedicated environment on `sparkmain` (`/home/sparkmain/bench_env/`).

### Run Standard Concurrency Sweep

```bash
python3 scripts/run_guidellm_suite.py \
  --profile concurrent \
  --streams 1,4,8,16,32 \
  --prompt-tokens 2048 \
  --output-tokens 256 \
  --duration 45 \
  --out-dir results/20260828-guidellm-concurrency-sweep
```

Outputs are automatically saved as:
- **`report.json`**: Machine-readable token statistics and percentiles.
- **`report.html`**: Interactive visualization of latency vs. concurrency curves.
