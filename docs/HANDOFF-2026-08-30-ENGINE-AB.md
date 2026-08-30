# Handoff — 2026-08-30, engine A/B arm 1 (eugr/spark-vllm-b12x)

> **⚠️ SUPERSEDED the same day by
> [HANDOFF-2026-08-30-EVENING-KSWEEP.md](HANDOFF-2026-08-30-EVENING-KSWEEP.md).**
> Read that one for current cluster state. Two things below are now WRONG:
> - the **"c=16 scheduling cliff"** attributed to speculative draft slots was mostly
>   **JIT contamination** — persistent kernel caches alone took c=16 aggregate
>   133.9 → 198.8 and TTFT 7000ms → 1755ms at identical nst=5. Every throughput
>   number in this file was taken under `--no-cache-dirs` and is a **LOWER BOUND**,
>   not engine capability.
> - the planned **nst=2** sweep point is **impossible**: the checkpoint sets
>   `dspark_block_size: 5` and nst<5 is rejected as producing incorrect output.
>   The sweep ran {5,7}; nst=5 won every cell.
>
> The correctness results, config deltas, and traps below remain valid.

Written for a fresh conversation with no prior context. Supersedes
[HANDOFF-2026-08-29-EVENING.md](HANDOFF-2026-08-29-EVENING.md) for cluster state.

---

## 1. Cluster state — READ FIRST

**The anemll engine is RETIRED as the serving engine (user decision 2026-08-30).** It stays
in the repo as the reference arm; every measurement row before 2026-08-30 carries
`engine=anemll-v0.25.1`. Its history is also frozen in the separate public repo
`colonel-otto/3x-dgx-spark-deepseek-v4-v0.25-anemll-baseline`.

- `dsv4.service`: **disabled, inactive**. Will not start on reboot. Do not start it while the
  eugr engine holds the GPUs.
- Overnight jobs **disabled and removed from the schedule**: `dsv4-prestart.timer` (01:50) and
  `issue22-deep-cells.timer` (02:00) — both were transient units; `systemctl list-timers`
  shows neither. The issue-22 deep cells are therefore NOT running tonight.
- **The eugr engine is UP** on all three nodes: container `vllm_node`, TP=3, 1M context,
  endpoint `http://<sparkmain wired mgmt IP>:8000`, served name `deepseek-v4-flash-eugr-ab`.
  It runs under a `nohup`'d `run-recipe.py` on sparkmain (log `~/eugr-ab-launch7.log`); it
  is NOT a systemd service yet — a reboot leaves the cluster with no engine.
- **The LAN gateway route is LIVE as of 2026-08-30 evening** (it was dead when this
  file was written, and briefly treated as out of scope). bigdog's LiteLLM (:4000)
  and the manifest (:8771) now resolve DSv4 to sparkmain **:8100**. It is not a
  prerequisite for benchmarking the engine, but it IS what LAN clients use — see the
  evening handoff.
- Weights: official `deepseek-ai/DeepSeek-V4-Flash-0731` @ 7872f01b at `~/dsv4/hf-…` on each
  node (156G, per-node home paths differ). The "abliterated" in old served names is legacy
  labeling; the weights are the official checkpoint and always were.

## 2. Preliminary results (arm 1) — see results/20260830T194550Z-engine-ab-eugr/

Correctness gate PASSED on the byte-identical 2-node-repo suite: quick gate 7/7, tool
battery 6/7 (`forced_choice` emitted valid JSON — tool-choice semantics, not garble),
deep-context 8/8, garble ALL CLEAN, RULER-lite 16/16 incl. 262K. The image's native
virtual-TP (heads 64→72, o_groups 8→9, zero-filled pad slabs and pad-head attn_sink) is
CORRECT at TP=3. **Our `apply_tp3_patch.py` is NOT applied and must not be.**

| c | anemll tp3-seqs16 decode / agg | eugr decode / agg | Δ agg |
|---|---:|---:|---:|
| 1 | 80.4 | 82.1 (warm; cold 65.4 under JIT compiles) | parity |
| 4 | 42.8 / 115.2 | 54.4 / 162.7 | **+41%** |
| 8 | 28.2 / 143.6 | 33.3 / 171.7 | **+20%** |
| 16 | 18.4 / 161.0 | 16.0 / 133.9 (best trial 161.6) | −17% |

The c=16 loss is a **scheduling cliff** (TTFT 1.9s at c=8 → 7.0s at c=16; compile-miss
counter frozen, so steady-state); startup warns `max_num_scheduled_tokens is set to 8128
based on the speculative decoding settings … decrease num_speculative_tokens or
max_num_seqs`. Config deltas vs anemll, all recorded per row: dspark nst=5 probabilistic
(anemll MTP K=2), kv fp8 (anemll nvfp4_ds_mla), V2 model runner, `--no-cache-dirs`.
DSpark acceptance at depth 5 was healthy (mean accepted 4.7–4.9, ~75%), so deeper drafts
are not being wasted — the budget is the constraint. KV: 2.42M tokens (fp8) vs 3.59M.
Weight load 78s (InstantTensor) vs 199s. Boot to serving ≈ 8 min with cold kernel caches.

**Not yet a same-day matched A/B** — the anemll engine was down; its side is the
2026-08-21 rows. Rows: `config_id=eugr-tp3-seqs16-dspark5`.

## 3. Next steps, in order (one variable each; protocol in ENGINE-AB-3NODE.md)

1. **Persist kernel caches.** Keep `--no-cache-dirs` to suppress the launcher's broken
   `$HOME`-relative worker mounts, then pre-create uniform reboot-persistent directories
   on every node and pass them explicitly:
   `-v /opt/eugrcache-vllm:/root/.cache/vllm
   -v /opt/eugrcache-flashinfer:/root/.cache/flashinfer -v /opt/eugrcache-triton:/root/.triton
   -v /opt/eugrcache-tilelang:/root/.tilelang`. Do not use `/tmp`: systemd-tmpfiles wipes it
   on reboot. Confirm the `cute.compile disk-cache-miss` count stops growing across boots.
2. **Speculative-depth sweep on this engine** — the open question "K=2 or bump to 5/7?":
   nst ∈ {2, 3, 5, 7} × c ∈ {1, 4, 8, 16}, same harness, median-of-≥5 on single-stream
   cells (33% noise band). Also the alternative lever at nst=5: `max_num_batched_tokens
   16384` — a KV trap on anemll (−43% KV, no gain), but the engine explicitly recommends
   it here and fp8 KV prices differently; measure, do not assume. Expect the winner to
   depend on concurrency exactly as it did on anemll (K=2 won at concurrency, K=5 single).
3. **Make eugr the service**: a systemd unit wrapping `run-recipe.py` with the working
   flags (or eugr's own launcher teardown), with `ExecStopPost` teardown and a `docker ps`
   check on all nodes — carry over the leaked-container lesson.
4. **Remaining A/B cells**: 131K-context decode, the code-brief / dense-prose prompt-effect
   pair, deep-concurrency 4×200K, then a proper fabric-gated run and — if ever wanted — a
   matched same-day A/B by booting anemll back for one session.
5. kv dtype: try `nvfp4_ds_mla` on this build to remove the KV-capacity delta, if supported.

## 4. Traps learned today (all in troubleshooting.md)

`--dry-run` validates only command construction; hosts.json Spark IPs are the WIFI
addresses (wired mgmt = enP7s7); Docker rejects symlink bind sources; the launcher expands
head `$HOME` on workers; its mesh NCCL env does not reach the container (inject the proven
`tp3.env` values); recipe 0.85 mem-util fails the startup check (0.82); never relaunch
within a minute of a failed attempt (teardown race, silent death); engine output streams to
the launcher log, not `docker logs`; JIT compile misses contaminate throughput until the
counter freezes.
