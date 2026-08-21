# Reproduction methodology

## Comparison matrix

Run all three configurations:

1. TP=2 over RoCE: production baseline.
2. TP=3 over TCP/Socket: transport control.
3. TP=3 over RoCE: target configuration.

Change only the node/TP count and explicitly named transport variables. Keep checkpoint,
container, patch state, serving profile, prompt, sampling and harness revision fixed.

## Single-stream protocol

- Greedy decoding (`temperature=0`).
- Fixed prompt stored with the artifact bundle.
- Request 256 output tokens.
- One excluded warm-up request.
- At least five measured requests; ten is preferred.
- Report every sample plus median, minimum, maximum, p50 and p95.
- Derive completion-token count from API `usage`; do not count SSE events because MTP
  can emit several tokens per event.

TTFT is the time from request submission to the first content-bearing stream event.
Decode throughput is completion tokens divided by the interval from first content to
the final event. Also retain end-to-end latency so readers can recompute alternatives.

## Concurrency protocol

Sweep `C=1,2,4,8,16` with the same prompt and output length. Report:

- aggregate output tokens per wall-clock second;
- per-request decode throughput;
- TTFT p50/p95;
- end-to-end latency p50/p95;
- completed, failed and queued request counts.

Do not label a wave metric as steady-state server throughput if the wave waits for its
slowest request. State the calculation explicitly.

## Long-context protocol

Use reproducible generated prompts at approximately 1k, 56k, 75k, 256k and 400k
tokens. Store prompt generator, random seed and tokenizer revision. For each length:

- perform needle retrieval at early, middle and late positions;
- report prefill throughput and TTFT separately from decode;
- capture peak memory and KV allocation;
- test concurrent long-context requests only within the configured `MAX_NUM_SEQS`.

This is necessary to demonstrate the practical benefit of the larger TP=3 KV pool;
short-prompt decode alone does not establish long-context capacity.

## Fabric protocol

Capture `NCCL_DEBUG=INFO` during first validation and require `NET/IB`. Run:

- `all_reduce_perf` across small and medium payloads, relevant to decode collectives;
- `all_gather_perf` at larger payloads, including NVIDIA's 16 GiB example;
- port transmit/receive counters immediately before and after a known workload.

Record counter deltas rather than publishing a counter snapshot without a time window.

## Artifact manifest

Every result directory should contain:

```text
manifest.json
rendered-compose.yaml
environment-redacted.txt
startup-rank0.log
startup-rank1.log
startup-rank2.log
benchmark-raw.jsonl
benchmark-summary.json
correctness.json
nccl-info.log
rdma-counters-before.json
rdma-counters-after.json
nccl-tests.csv
sha256sums.txt
```

See [`../artifacts/README.md`](../artifacts/README.md) for the schema and redaction rule.
