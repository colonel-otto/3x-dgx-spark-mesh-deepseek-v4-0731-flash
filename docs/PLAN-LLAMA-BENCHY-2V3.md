# Plan — llama-benchy as an independent 2-node vs 3-node arm

**Status:** plan only. Nothing measured yet. Tool installed and smoke-tested; the
measurement run has not been launched.

**Purpose.** Reproduce the 2v3 node-count question on a **third-party harness** that the
DGX Spark community actually uses, so the result stops depending on our own harness. Our
matched run ([`RESULT-2V3-MATCHED-2026-08-30.md`](RESULT-2V3-MATCHED-2026-08-30.md)) found
three nodes faster on every workload. If an independent tool disagrees, that is a finding.
If it agrees, the claim stops being self-certified.

**Explicitly out of scope: 1 node.** Not tested, not planned. The official vLLM recipe
states a single GB10's 128 GB unified memory is below the DeepSeek-V4-Flash checkpoint
footprint, so a 1-node arm of this model does not exist. This plan is 2 vs 3 only.

---

## 1. Why this tool

[`eugr/llama-benchy`](https://github.com/eugr/llama-benchy), already cloned at
`~/llama-benchy` on sparkmain (`e9be344`).

It is the right third-party choice for four specific reasons, not merely availability:

| Property | Why it matters here |
|---|---|
| Benchmarks any **OpenAI-compatible endpoint** | Points at the live vLLM server. No re-serving, no GGUF conversion, no second inference engine to confound the comparison. |
| **`--exact-tg`** sends `min_tokens` + `ignore_eos` | This is our Requirement 2 (asserted output window) implemented natively. The harness defect that caused our 25-token collapse cannot recur here. |
| **Handles MTP chunks correctly**, and prompts come from a real Gutenberg book rather than random tokens | Speculative decoding is measured honestly. Random-token prompts distort draft acceptance — which is precisely why our own numbers are prompt-sensitive (a measured 1.65x swing). |
| Built-in **coherence test** after warmup | The automated equivalent of the TP=3 "silently serves fluent nonsense" trap that `patch.md` warns about. |

It also supports `--runs` with mean ± std, `--warmup-runs` per shape, and separates
aggregate from per-client throughput at concurrency — the aggregate-metric trap already
documented in this repository.

## 2. Tool verification already done

- Installed to `~/llama-benchy/.venv` (no `uv` on the box; `python3 -m venv` + `pip -e .`).
- Smoke test against the live TP=3 endpoint **passed**: coherence test PASSED,
  `pp2048 = 1393 t/s`, `tg32 = 44.5 t/s`.
- **Tokenizer defect found and fixed.** On the first run llama-benchy could not fetch the
  DeepSeek tokenizer from HuggingFace (auth failure) and **silently fell back to `gpt2`**.
  Token counts then become approximations, making every `t/s` figure wrong. The corpus
  measured 159,385 tokens under `gpt2` versus **142,813** under the real tokenizer — an
  **11.6 % discrepancy** that would have propagated into every published number.

  Fixed by extracting the real tokenizer out of the container to `~/dsv4-tokenizer/` and
  passing `--tokenizer`. **This flag is mandatory on every invocation below.** A run
  without it is not publishable.

## 3. Fairness rules — the same discipline as the matched run

1. **Node count is the only variable.** Both arms at `MAX_NUM_SEQS=32`,
   `MTP_NUM_TOKENS=2`, `GPU_MEMORY_UTILIZATION=0.835`, `MAX_NUM_BATCHED_TOKENS=8192`,
   `MAX_MODEL_LEN=1048576`, `LONG_PREFILL_TOKEN_THRESHOLD=1024`,
   `DSPARK_MAX_INFLIGHT_PREFILLS=2`, `KV_CACHE_DTYPE=nvfp4_ds_mla`,
   `VLLM_PREFIX_CACHE_RETENTION_INTERVAL=4096` — asserted against the **live engine**
   before measuring, as `run_matched_2v3_powered.sh` already does.
2. **`--tokenizer ~/dsv4-tokenizer` on every invocation.** Non-negotiable; see §2.
3. **`--exact-tg`**, so the output window is asserted rather than hoped for.
4. **Fabric gate** before each arm, artifact committed.
5. **`open-webui` stopped** for the duration (Requirement 5).
6. **Cool to ≤70 °C before each arm**, sample clocks every 5 s. GB10 clocks cannot be
   locked, so thermal state is equalised instead.
7. **Record the software stack** — driver 580.173.02, CUDA 13.0, NCCL 2.28.9, vLLM build.
   Driver is a first-class variable on GB10 (a ~3.5x regression is documented between two
   580.x releases), so a number without it is not externally comparable.
8. **Both arms in one session**, same night, same fabric state.

## 4. The runs

Two arms, each with a depth sweep and a concurrency sweep, mirroring the axes our own
harness measured so the two can be checked against each other.

### Depth / prompt-processing sweep (per arm)

```bash
cd ~/llama-benchy && .venv/bin/llama-benchy \
  --base-url http://127.0.0.1:8100/v1 \
  --model deepseek-v4-flash-0731 \
  --tokenizer ~/dsv4-tokenizer \
  --pp 2048 --tg 256 --exact-tg \
  --depth 0 8192 32768 131072 \
  --runs 10 --warmup-runs 3 \
  --latency-mode generation \
  --format json --output <ARM>-depth.json
```

### Concurrency sweep (per arm)

```bash
cd ~/llama-benchy && .venv/bin/llama-benchy \
  --base-url http://127.0.0.1:8100/v1 \
  --model deepseek-v4-flash-0731 \
  --tokenizer ~/dsv4-tokenizer \
  --pp 8192 --tg 256 --exact-tg \
  --depth 0 --concurrency 1 4 8 16 \
  --runs 10 --warmup-runs 3 \
  --latency-mode generation \
  --format json --output <ARM>-concurrency.json
```

`--runs 10` rather than 30: llama-benchy reports mean ± std per cell, and this is a
**cross-harness agreement check**, not the primary result — the primary result already has
n=30. If a cell disagrees with our harness by more than its own standard deviation, raise n
on that cell and re-run rather than reporting a marginal difference.

**262K is deliberately excluded.** At roughly 173 s per rep it would dominate the runtime
of a confirmatory arm. Our own harness already covers it at n=12 with Cliff's δ = 1.000.

## 5. Pre-registered expectations

Stated before running, so agreement cannot be claimed retroactively.

| # | Expectation | What would falsify it |
|---|---|---|
| L1 | llama-benchy reproduces the **direction** of every matched-harness result: three nodes faster at every depth and every concurrency | Any cell where two nodes lead by more than that cell's own std |
| L2 | Absolute `tg` t/s will **not** match our numbers exactly — different prompt corpus, and different depth semantics | n/a — expected, and the reason only *direction* and *ratio* are compared |
| L3 | The 2v3 **ratio** agrees within about 5 percentage points of ours (+17 % to +20 % at 8K–131K) | A ratio outside ±5 pp, implying one harness measures something the other does not |

**L2 is the load-bearing caveat.** `llama-benchy --depth N` prefills N tokens of *cached
context* and then measures `pp`/`tg` on top of it. Our `decode_depth_sweep.py` sends an
N-token prompt with **no** caching. These are different measurements.
**Cross-harness absolute numbers must never be compared — only the 2v3 ratio computed
within each harness.**

## 6. Cost

| Phase | Time |
|---|---|
| TP=2 bringup (cold start ~30 min) | 30 min |
| TP=2 depth + concurrency | ~45 min |
| Restore TP=3 (cold start) | 30 min |
| TP=3 depth + concurrency | ~45 min |
| **Total** | **~2.5 h**, cluster down for most of it |

## 7. What this buys

- A 2v3 result on a harness **we did not write**, from a tool the GB10 community publishes
  with — so the claim stops being self-certified.
- A **cross-check on our own harness**. If both agree, the +17–20 % is corroborated. If they
  disagree, we have found a harness artifact in one of them, which is worth more than the
  benchmark itself.
- Numbers on the **same axis as Spark Arena** and the wider llama-benchy community, which is
  the only way our results can be read against anyone else's.

## 8. What it does not buy

- **Not a 1-node comparison.** Impossible for this model (see scope, above).
- **Not comparability with published Spark numbers** at different quantization, context
  depth, or driver version. Those remain uncomparable regardless of harness.
- **Not a replacement for the matched run.** This is confirmatory, at lower n.
