# Continuation report — eugr engine A/B, 2026-08-30

> **⚠️ Point-in-time record, written BEFORE the sweep ran. Superseded by
> [HANDOFF-2026-08-30-EVENING-KSWEEP.md](HANDOFF-2026-08-30-EVENING-KSWEEP.md).**
> Kept because its state-vs-claims separation is a useful audit of what was
> actually established at that moment. Three statements below were overtaken by
> measurement the same evening:
> - "The c=16 result is attributed to the engine's 8128 scheduled-token budget
>   interacting with depth-5 draft slots" — **retracted**. Mostly JIT
>   contamination: persistent kernel caches alone took c=16 aggregate 133.9 → 198.8
>   and TTFT 7000ms → 1755ms at identical nst=5.
> - "eugr is a manual `nohup` launcher … listening on port 8000" — it is now
>   `eugr.service` (enabled + active) on **:8100**, serving both names.
> - "The LAN gateway is outside the scope" — the route is now restored and verified
>   end-to-end (its LiteLLM block had been deleted, not merely misconfigured).

This report converts the previous agent transcript into an actionable state
record. It is intentionally conservative: repository evidence is separated
from claims that still require a live-cluster check.

## Executive summary

The eugr/spark-vllm-b12x image successfully served the official DeepSeek-V4
Flash 0731 checkpoint at TP=3 on three DGX Spark nodes. Its native B12X
virtual-TP padding passed the correctness suite, so the repository's own TP=3
padding patch must not be applied on this engine.

The first throughput arm is promising but not a final A/B: it used eugr with
dspark speculative depth 5 and fp8 KV, while the comparison rows are older
anemll results with MTP K=2 and nvfp4 KV. The measured eugr result was roughly
single-stream parity, +41% aggregate at concurrency 4, +20% at concurrency 8,
and −17% at concurrency 16. The c=16 result is attributed to the engine's
8128 scheduled-token budget interacting with depth-5 draft slots, not to a
fabric failure or poor draft acceptance.

The live cluster is currently in an operationally incomplete state: eugr is a
manual `nohup` launcher on `sparkmain`, listening on port 8000 under the eugr
model name. The LAN gateway is outside the scope of bringing up or measuring
this container and is intentionally not a prerequisite here.

## What the previous agent completed

- Commit `ba1d018` added the generated-recipe boot script, sweep driver, and
  vendored arm-1 harness.
- The boot script was dry-run validated for port 8100, both served names,
  speculative depth, batched-token budget, and exactly one speculative config.
- The live fabric precondition was reported as 12/12 links active at 200 Gb/s.
  This is useful operational evidence, but the arm-1 result bundle is marked
  `fabric_gate: ABSENT`; it is not a publishable fabric-gated comparison.
- The service unit and service helpers had been created but
  were untracked because stopping the running engine was blocked by the
  execution guard.

## Evidence-backed state

| Item | Evidence | Interpretation |
|---|---|---|
| Correctness | `results/20260830T194550Z-engine-ab-eugr/README.md`: quick 7/7, deep context 8/8, garble ALL CLEAN, RULER 16/16, tools 6/7 with valid JSON | Native eugr TP=3 path is usable; the tools difference is semantics, not garble |
| Engine identity | `engine-config.txt`: vLLM dev build, TP=3, 1,048,576 max sequence, fp8 KV, dspark `num_spec_tokens=5`; image digest `7dc02f16…` | Reproduce with the exact image/config, not the anemll defaults |
| Throughput | Result README and `benchmarks/measurements.csv`: 82.1 single-stream, 162.7 aggregate c=4, 171.7 c=8, 133.9 c=16 | Preliminary arm-1 numbers; c=4/c=8 are the useful signal |
| JIT contamination | Result README: 20 post-start `disk-cache-miss` events; c=1 moved from 65.4 cold to 82.1 warm | Cache persistence must precede the next measurement |
| c=16 cliff | Result README: TTFT about 1.9 s at c=8 versus 7.0 s at c=16; startup warning cites `max_num_scheduled_tokens=8128` | Sweep speculative depth and batched-token budget instead of assuming a hardware regression |
| Cache-path root cause | `docs/troubleshooting.md`: launcher default mounts are `$HOME`-relative and head `$HOME` is shipped to workers | Keep `--no-cache-dirs` and add uniform absolute `/opt/eugrcache-*` bind mounts |
| Local endpoint state | Current handoff: live eugr endpoint is `sparkmain:8000` with `deepseek-v4-flash-eugr-ab` | Verify locally on sparkmain; no external gateway is required |
| Service state | Current handoff and untracked `eugr.service`: eugr is not yet systemd-managed | Install only after the live manual launcher is stopped |

## Correct next sequence

1. On `sparkmain`, first identify the current manual launcher's exact PID and
   container state. Install the artifacts with
   `scripts/eugr-ab/install-service.sh`; it enables but intentionally does not
   start the unit. Then stop that exact manual launcher and remove `vllm_node`
   on all three nodes. Verify `docker ps -a` is clean on every node. Do not
   start the old `dsv4.service` while eugr owns the GPUs.
2. Start `eugr.service` with `EUGR_NST=2` and `EUGR_MNBT=8192` (the first sweep
   point). The boot script creates `/opt/eugrcache-{vllm,flashinfer,triton,tilelang}`
   on every node, generates a recipe, dry-runs its command, then launches on
   port 8100 under both names.
3. Verify `sparkmain` locally with `/v1/models` and one completion. No gateway
   host, manifest service, or external route is involved.
4. Run the fabric gate in the measured engine state and retain its JSON beside
   the new result bundle. A prior “12/12 links active” check is not a substitute
   for the repository's committed gate artifact.
5. Run the hardened sweep driver for each point: `nst ∈ {2,3,5,7}` and
   `c ∈ {1,4,8,16}`, plus `nst=5,mnbt=16384`. It now warms every concurrency,
   requires a frozen JIT-miss counter, uses 256-token completions, verifies
   exactly five trials, and checks the request-success delta for the expected
   145 measured requests per four-concurrency sweep.
6. Only after the clean K sweep, run the remaining 131K decode, code/prose
   prompt pair, deep 4×200K concurrency, and any matched same-day A/B that is
   still desired.

## Important interpretation limits

- “Correctness passed” does not mean the engines are a matched A/B. The anemll
  side was not live on the same day.
- Do not call the arm-1 numbers cold, cache-clean, or fabric-gated.
- Do not infer that K=5 is wasteful: observed depth-5 acceptance was about
  4.7–4.9 accepted tokens per draft, so the open question is scheduler cost
  versus single-stream benefit.
- Do not use `/tmp/eugrcache-*` for persistence; systemd-tmpfiles clears `/tmp`
  on reboot. `/tmp/dsv4` remains a deliberately recreated hardlink farm for
  the model bind mount.
- Do not use a second `--speculative-config` appended after the recipe. The
  launcher appends passthrough arguments; the generated recipe avoids duplicate
  JSON flags and records the exact sweep point.

## Handoff prompt for the next agent

> Read `docs/CONTINUATION-REPORT-2026-08-30-ENGINE-AB.md` and
> `docs/HANDOFF-2026-08-30-ENGINE-AB.md`. The local artifacts are ready, but
> the live eugr `nohup` engine still owns the GPUs and the gateway is still
> outside the scope of this work. First verify the live process/container state, then stop the launcher
> and clean all three nodes only with explicit approval. Install the eugr
> service artifacts, boot nst=2/mnbt=8192 on port 8100 with both model names,
> run the local endpoint and fabric gates, and start the hardened sweep. Preserve the
> old anemll service as disabled reference state and do not apply the TP=3
> padding patch.
