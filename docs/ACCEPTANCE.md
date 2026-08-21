# Acceptance gates

## Gate A — Baseline is frozen

- baseline run exists under `results/`
- environment snapshot exists for both current Sparks
- model revision and vLLM/container identity are recorded
- baseline endpoint completes benchmark without request failures

## Gate B — Three-node fabric

- all 3 management IPs reachable by passwordless SSH
- both CX-7 ports visible/up as expected on each Spark
- NVIDIA three-node NCCL ring command completes
- `#wrong = 0`
- bandwidth is captured for later comparison/debugging

## Gate C — Distributed runtime

- either multi-node `mp` reaches all three ranks, or Ray reports exactly 3 alive nodes
- exactly one GPU resource per Spark participates
- execution environments are identical

## Gate D — DeepSeek PP=3

- same DeepSeek-V4-Flash-0731 revision as baseline
- `TP=1`
- `PP=3`
- `VLLM_PP_LAYER_PARTITION=14,15,14`
- speculative decoding disabled
- API becomes healthy without distributed initialization errors

## Gate E — Correctness

- zero HTTP/request failures in the standard suite
- 100% needle retrieval at 2K, 8K and 32K targets
- extend to 64K and 128K before claiming long-context improvement

## Gate F — Before/after report

Record, do not guess:

- median TTFT
- median per-request decode tokens/sec
- median aggregate output tokens/sec at concurrency 1, 3 and 6
- actual prompt-token counts
- success/needle accuracy
- environment difference

There is deliberately no hard speed threshold. The experiment determines whether three Sparks improve the workload you actually care about.
