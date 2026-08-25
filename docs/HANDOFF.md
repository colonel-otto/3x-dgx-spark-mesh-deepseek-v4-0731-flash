# Handoff — DeepSeek-V4-Flash 3×DGX Spark, as of 2026-08-24

Read this first if you are picking up this work. It states what is running, what is
settled, what is open, and exactly how to run the next three measurements.

---

## 1. What is running right now

**Verified live 2026-08-24 22:5x UTC.** The cluster is healthy and serving.

| | value |
|---|---|
| Endpoint | `http://192.168.1.223:8100` (LAN), `http://localhost:8100` on the head |
| Model id | `deepseek-v4-flash-0731` |
| Weights | `/models/dsv4-abliterated` — DeepSeek-V4-Flash-0731, abliterated |
| Image | `ghcr.io/anemll/dspark-vllm-gx10:0.1.1` |
| vLLM | `0.25.2.dev0+g752a3a504.d20260714` |
| Topology | TP=3, PP=1, NNODES=3, `flashinfer_b12x` MoE |
| `MAX_MODEL_LEN` | **1,048,576** |
| `MTP_NUM_TOKENS` | **5** |
| `MAX_NUM_SEQS` | 16 |
| `GPU_MEMORY_UTILIZATION` | **0.80** |
| `kv-cache-dtype` | `nvfp4_ds_mla` |
| CUDA-graph capture | 96 (`16 × (5+1)`), 1.24 GiB / 15 s |
| GPU KV cache | **4,480,480 tokens** (4.3x concurrency at full context) |

### The three nodes

| rank | host | note |
|---:|---|---|
| 0 | `sparkmain` | head — **the only node that serves**; workers emit nothing |
| 1 | `spark1` (`gx10-e146`) | worker |
| 2 | `spark2` (`gx10-6b41`) | worker. **`spark2` and `spark-sep` are the same machine** — two aliases, one host. Do not count them as four nodes. |

### Memory headroom — read before raising anything

```
sparkmain: total 121Gi  used 117Gi  available  3Gi
spark1:    total 121Gi  used 117Gi  available  4Gi
spark2:    total 121Gi  used 117Gi  available  4Gi
```

Roughly **4 GiB available per node**. Any change that grows CUDA-graph capture or the
KV pool has to fit in that. This is the single most important constraint on issue #10.

---

## 2. How to operate the cluster

### `dsv4.service` now starts the 3-node cluster — FIXED 2026-08-24

It previously started a **2-node TP=2** topology from `config/head.env`. It now drives
`config/tp3.env` on all three ranks:

- `~/bin/dsv4-service-start` — waits for **both** workers' SSH (150 × 4 s; a Spark can
  take 10–20 min to open port 22 after a cold boot, which is normal), **verifies
  `tp3.env` is byte-identical across all three ranks**, then starts workers → 15 s →
  head, and waits for `/health`.
- `~/bin/dsv4-service-stop` — head first, then workers, best-effort so an unreachable
  node cannot block teardown. `ExecStop` previously called the 2-node `dsv4 down`.
- Unit: `TimeoutStartSec` raised 900 → 2400 (900 could not cover a real cold boot).

The config-verification step is the valuable part: a mismatch between ranks otherwise
hangs startup **forever with no error**, and this turns that into a clear failure in two
seconds.

Old versions kept as `~/bin/dsv4-service-start.bak-2node-*` and
`/etc/systemd/system/dsv4.service.bak-2node-*`.

> **`~/bin/dsv4` and `~/dsv4/dsv4` are still the old 2-node launcher** (they are the
> same file — one is a symlink). The service no longer calls them, but do not invoke
> them by hand on this cluster.

**Status: `active` but `disabled`** — it will not auto-start on boot. Enabling it is a
deliberate decision that has not been made; auto-start on boot sits close to the
standing "no autonomous recovery actions" rule.

### Config location

`~/localai/dspark-vllm-gx10/config/tp3.env` on **each of the three nodes**, separately.

> **Trap:** a mismatched `MAX_MODEL_LEN` (or any parallelism flag) between nodes hangs
> startup **forever with no error message**. Always edit all three, then verify all
> three before restarting.

### Restart procedure — workers first, then head

```bash
# 1. Edit config on ALL THREE ranks, then verify:
for h in sparkmain spark1 spark2; do ...grep -E 'MAX_MODEL_LEN|MAX_NUM_SEQS|MTP_NUM_TOKENS|GPU_MEMORY' tp3.env; done

# 2. Down: head first, then workers
ssh sparkmain "cd ~/localai/dspark-vllm-gx10 && COMPOSE_DISABLE_ENV_FILE=1 \
  docker compose -p dspark-vllm-gx10 --env-file config/tp3.env -f docker-compose.yml down"
# then the same on spark1 and spark2

# 3. Up: WORKERS FIRST (rank 1, rank 2), wait ~15s, THEN head (rank 0)
#    Rank 0 blocks waiting for the workers to join the collective.

# 4. Wait for health. Cold start is ~7 min (375-425s observed).
curl -sf http://localhost:8100/health
```

A full restart cycle is **~12 minutes** end to end. Budget for it.

> **Warn the user before restarting.** VS Code Remote-SSH forwards these ports and its
> session drops when they go away.

### Verify a restart took effect

```bash
curl -s http://localhost:8100/v1/models | python3 -c \
  'import json,sys;d=json.load(sys.stdin)["data"][0];print(d["id"],d["max_model_len"])'

docker logs dspark-vllm-gx10-vllm-dspark-1 2>&1 | grep -oE \
  "num_spec_tokens=[0-9]+|max_cudagraph_capture_size': [0-9]+|GPU KV cache size: [0-9,]+"
```

Never assume the config applied. Read it back from the engine.

---

### The 2-node fallback is intact

A complete **TP=2 profile** survived the move to three nodes and is verified coherent as
of 2026-08-24: `config/head.env` on the head plus `config/worker.env` on worker 1, with
worker 2 simply unused. It runs on a separate fabric path and needs **no padding patch**
(64 heads divide by 2 cleanly), which makes it a genuinely simpler fallback.

Neither file sets `TP_SIZE` or `NNODES` — compose defaults them to 2
(`${TP_SIZE:-2}`, `${NNODES:-2}`). **That absence is load-bearing, not a bug.**

Keep it for **availability**, not performance: TP=3 currently *leads* the published
2-node recipe on decode (peak 91.1 vs 84.3, mean 76.0 vs 67.6). The real cost of falling
back is KV capacity — roughly 1.86M tokens against 5.44M.

Full rollback procedure in the operator's `docs\dsv4-2node-fallback.md`.

## 3. What is settled — do not re-litigate

| Finding | Status |
|---|---|
| `MTP_NUM_TOKENS=5` beats `4` | **Settled 2026-08-24** with the matched control (seqs held at 16). +8–13% single-stream on structured/code. Accepted draft length rises ~1 full token while acceptance % stays flat. Prose is the one workload preferring 4. |
| 1M context is free | **Settled.** KV pool *grew* 3.57M → 5.44M tokens; aggregate unchanged within spread. Never run this cluster at 460,800 again. |
| Patch 4 (shared expert) | **Does not apply** to vLLM 0.25.2 — the generic substring mapping already catches `.shared_experts.w1/.w3`. Verified three ways. Applying it would be redundant. |
| KV dtype is not a speed lever | fp8 vs nvfp4 measured **identically** upstream (41.4 vs 41.5). Do not switch KV dtype for throughput reasons. |
| The 60 tok/s "ceiling" | **False for this cluster.** We measure 93.8 / 92.3 / 86.1 tok/s single-stream. |
| `MAX_NUM_SEQS=32` | **Rejected 2026-08-24.** Crashes the engine at sustained cc=32 (NCCL allgather stall); cc=16 also regressed. Keep 16. |
| `NCCL_IB_MERGE_NICS` | **No-op 2026-08-24.** NCCL already merges both HCAs by default; measured identical both ways. |
| GB10 GPUDirect RDMA | **Absent — architectural.** All inter-node collectives host-stage at ~0.5 GB/s. This is the prefill ceiling. |
| `GPU_MEMORY_UTILIZATION=0.80` | **Shipped 2026-08-24.** +14% prefill at 78K vs 0.85, decode unaffected. Costs 17% of KV pool. |
| TP=3 attention head padding | **50% waste is real, but NOT prefill-limiting.** Kernel widths {8,16,32,64,128}; 24 heads/rank snap to 32. But TP=2 and TP=3 measure IDENTICAL prefill at 8K (1,081 both), so this is not the prefill bottleneck. |
| TP=2 as a prefill fallback | **Rejected 2026-08-25.** Same prefill as TP=3, but decode cc=1 drops 82.4 -> 70.1 and KV 4.48M -> 1.82M. TP=3 is correct. |
| Prefill gap cause | **Localised 2026-08-25 to a fixed ~470 us/token communication cost.** Flat across a 32x prompt-size range. Every software variable AND GPU compute/clocks/bandwidth eliminated by measurement. |
| GB10 GPU health | **Verified healthy.** 93.3 TFLOP/s bf16, 236 GB/s memory, 2,470 MHz vs a 2,418 MHz application-clock spec. An earlier claim of '82% of rated' was WRONG — 3,003 MHz is a hardware ceiling, not the operating spec. |
| Upstream harness on our cluster | **Reproduces our numbers.** anemll's benchmark_prefill.py gives 1,075 tok/s @ 8K (server timer, 0.3% agreement with client). Not a measurement artifact. |
| Aggregate throughput as an MTP signal | **Useless** — every level sits inside its own run spread. Use the acceptance counters. |

Full detail and evidence: [`MTP5-1M-AND-UPSTREAM-COMPARISON.md`](MTP5-1M-AND-UPSTREAM-COMPARISON.md).
Historical traps and falsified experiments: [`TP3-TUNING.md`](TP3-TUNING.md).

---

## 4. Measurement discipline — read before running anything

These are not style preferences. Each was learned by getting a wrong answer first.

1. **Warm up until two sweeps agree.** A cold sweep reads **~40% low** here — cc=16
   measured 206–222 tok/s cold against a steady-state 340–375. One warm-up pass is not
   enough after any change that alters graph capture. Discard warm-up output entirely.

2. **Confirm the endpoint is idle before every run.**
   ```bash
   curl -s http://localhost:8100/metrics | grep -E '^vllm:num_requests_running\{'
   ```
   One in-flight request from elsewhere skews a level badly.

3. **Change one variable at a time.** The MTP question stayed open for weeks because an
   earlier sweep moved `MTP_NUM_TOKENS` and `MAX_NUM_SEQS` together. If you must change
   two, you have produced no answer about either.

4. **Prefer acceptance counters over throughput for spec-decode questions.** At
   temperature 0 with a fixed prompt they are near-deterministic (identical to 2 decimals
   across reps), while aggregate throughput spreads 12–34%.

5. **Measure at the head node only.** Under TP the workers serve nothing. Summing
   per-node double-counts.

6. **Run upstream harnesses unmodified.** Every cross-repo number in this repo comes from
   running *their* script against our endpoint — never from quoting our harness against
   their published table. Those measure different things.

7. **State what you did not test.** An unexplained gap recorded honestly beats a
   fabricated cause.

8. **Launch long remote runs with `nohup` and redirect to a file.** A benchmark that
   takes more than a few minutes must survive its launcher dying:
   ```bash
   ssh sparkmain "nohup python3 /tmp/thing.py args > /tmp/thing.log 2>&1 &"
   ```
   On 2026-08-24 a ~25-minute run was launched through a plain `ssh` pipe. The local
   wrapper was torn down; the **remote process kept running** (found alive 17 minutes
   later, still mid-request) but its stdout went to a dead pipe, so every result it had
   produced was unrecoverable. Prefer small batches per invocation too, so a loss costs
   one depth rather than all of them.

   If a background task reports "stopped" with no completion record, check
   `pgrep -af <script>` on the remote host before assuming it never ran — and note that
   a killed client can leave an in-flight request on the server for minutes afterwards.

### The harnesses

| script | source | measures |
|---|---|---|
| `/tmp/bench_tp3.py` | `localaiguyy/...-3x-DGX-Spark` | concurrency sweep, aggregate + per-stream |
| `/tmp/bench_full.py` | `tonyd2wild/...-2x-DGX-Spark` | decode by content type, concurrency, prefill |
| `results/20260824-mtp5-1m/accept.py` | written here | tok/s + acceptance % + accepted draft length |

`/tmp` is not durable. Re-copy from `results/20260824-mtp5-1m/` or the upstream repos.

---

## 5. Next measurements — in priority order

> **START HERE: prefill parity NOT reached, but the cause is now a MEASURED HARDWARE
> ASYMMETRY: one fabric link runs 6.8x slower than the other.**
> Read [`PREFILL-MEASURED.md`](PREFILL-MEASURED.md) — all five addenda.
>
> **THE FINDING (maintenance window, vLLM stopped, 64 MiB allgather):**
>
> | pair | busbw |
> |---|---:|
> | sparkmain <-> spark2 (f0<->f0) | **4.64 GB/s** |
> | sparkmain <-> spark1 (f0<->f1) | **0.69 GB/s** |
> | all 3 ranks | **0.49 GB/s** |
>
> A TP collective is paced by its slowest link, which is why 3-rank sits below even the
> bad pair. All six ports negotiate 200,000 Mb/s with zero error counters.
>
> **NEXT ACTION — a targeted hardware test with a predicted outcome:** swap the
> sparkmain-f0 <-> spark1-f1 cable, or move that pair onto the unused `roceP2p1s0f*`
> HCAs. Predicted: 0.69 -> ~4.6 GB/s. Then re-run the prefill benchmark immediately.
>
> - **Shipped: `GPU_MEMORY_UTILIZATION` 0.80** (+14% prefill at 78K, decode unaffected).
> - **Fixed (NOT persistent — reverts on spark1 reboot):** duplicate `192.168.100.2/24`
>   on both of spark1's NICs, plus two stale routes. Correct hygiene; did NOT change
>   bandwidth or prefill. Make it persistent in spark1's network config.
> - **Ruled out by measurement:** `NCCL_DMABUF_ENABLE=1` + `NCCL_NET_GDR_LEVEL=5`
>   (0.49 GB/s either way), harness (byte-identical `token_pool_sha256` to upstream's
>   published run), prefix caching, TP=2 vs TP=3, head padding, vLLM version, image,
>   checkpoint, `MAX_MODEL_LEN`, `MTP_NUM_TOKENS`, seqs, gpu-mem, indexer chunk MB,
>   prompt content, GPU compute (93.3 TFLOP/s), clocks (2,470 vs a 2,418 spec — at
>   spec), memory bandwidth (236 GB/s of 273).
>
> **CAUTION:** `MTP_NUM_TOKENS=0` is invalid (service fails to start); use 1 as a floor.
> MTP=1 leaves prefill flat but **collapses decode to 46.7 tok/s**.
>
> **A method note:** an earlier probe flagged this same spark1 leg and I dismissed it
> using a TCP test (within 1.27x). TCP does not exercise the RDMA verbs path — wrong
> instrument. Use the NCCL allgather (`results/20260824-seqs32-nccl/agbench.py`), which
> is what vLLM actually uses.
>
> Cluster: **idle, healthy, serving** — TP=3, 1M context, MTP=5, seqs=16, 0.80,
> KV 4,604,327. Decode cc=1 82.6 tok/s, cc=16 341.9.

### 5a. `MAX_NUM_SEQS=32` → issue #10 — **CLOSED 2026-08-24: REJECTED**

> **Do not retry this.** It boots and serves, then kills the engine at sustained
> cc=32 — an `_ALLGATHER_BASE` stall on all three ranks with KV only 2.8% full and
> zero link errors. cc=16 also measured *worse* than the seqs=16 baseline
> (325.1 vs 374.2). KV cost was only -1.1%, so the memory risk this section warned
> about was not the problem. Full analysis:
> [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md).

<details><summary>Original scoping (kept for context)</summary>

**Why:** the one upstream number we cannot match. The 3-node repo reports 618 tok/s
aggregate at seqs=32 versus 431 at seqs=16 (+43%); we measure 374.2 at seqs=16.
Configuration, not silicon, has been the ceiling through three of their settings.

**The risk, concretely.** Capture size is *derived*:

```
capture = MAX_NUM_SEQS × (MTP_NUM_TOKENS + 1) = 32 × 6 = 192
```

That is **double** our current 96. Two things make this riskier for us than for them:
our `GPU_MEMORY_UTILIZATION` is 0.85 against their 0.80, and we run MTP=5 so the
multiplier is 6 rather than 6 at their k=5 — same multiplier, but from a higher base.
**We have ~4 GiB available per node.** Our capture at 96 cost 1.24 GiB; theirs at 192
cost 1.71 GiB. It plausibly fits, but it is not guaranteed.

**Steps:**
1. Set `MAX_NUM_SEQS=32` in `config/tp3.env` on all three ranks. Verify all three.
2. Restart (workers → head). Watch for OOM during capture specifically.
3. Confirm `max_cudagraph_capture_size: 192` and record the KV pool delta.
4. Warm at the concurrencies you will measure — **a capture-size change invalidates
   prior warm-up.** The 3-node repo found sweep 1 read 40% low after exactly this change,
   which would have hidden the entire gain. Repeat until two sweeps agree.
5. `python3 /tmp/bench_tp3.py --host localhost:8100 --concurrency 1,8,16,24,32,40 --runs 3 --json /tmp/seqs32.json`

**If capture OOMs:** lower `GPU_MEMORY_UTILIZATION` toward 0.78. **Do not drop
`MAX_MODEL_LEN` first** — that trades away context for no reason.

**Record:** aggregate at cc=32 with spread, whether saturation moves from cc=16 to
cc=32, the KV pool delta, and single-stream (expected flat — this buys aggregate, not
latency). **Rollback is a config revert plus one restart.**

</details>

### 5b. NVFP4 KV output quality → issue #12 — **single-request half DONE 2026-08-24**

**Result: clean through 463,792 prompt tokens.** Needles at 10/50/90% depth all recovered
exactly at every depth from 2K to 500K, with no garble of any kind. 500K is the depth the
upstream warning specifically aims at. **Keep `nvfp4_ds_mla`** — the case for switching is
currently zero on both axes (performance measured identical to fp8; quality shows no
degradation). Full writeup: [`KV-QUALITY-LONG-CONTEXT.md`](KV-QUALITY-LONG-CONTEXT.md).

**What remains open on #12** — the untested half, and the more likely failure mode:

1. **Concurrency.** Every measurement was a single request against an idle engine. The
   upstream warning pairs long context *with* concurrency ("clean output **under
   concurrency**"), and KV pressure is a plausible trigger this test never applied.
2. **Agentic structure.** Filler was prose, not tool calls / JSON state / multi-turn
   history. Upstream describes *"heavy agentic context"* specifically.
3. **Repetitions and temperature > 0.** One prompt shape, one seed, temp 0 — the most
   favourable case. A rare intermittent failure would not have appeared.

Run `results/20260824-kv-quality/kvquality.py` concurrently (several simultaneous
long-context requests) as the next probe. **Only if something fails** is an fp8 A/B at
matched context warranted — there is no effect to attribute otherwise.

Depths above ~464K were not measured: a 700K/900K run was started and deliberately
abandoned once 500K passed, since 500K is the depth the upstream warning targets.

<details><summary>Original scoping for this issue (kept for context)</summary>

**Why:** this is a **correctness** test, not a benchmark, and it is the one open risk to
the 1M context we just adopted. The 2-node repo warns 4-bit KV *"can collapse into salad
under long, heavy agentic context"* while fp8 stays clean.

**Do not switch KV dtype on throughput grounds** — that rationale is measured false
(§3). Only quality justifies it.

**Steps:**
1. Drive genuine agentic context toward 200K, then 500K, then beyond. Watch for
   multilingual / BOS "salad" and coherence collapse.
2. Determine whether any degradation correlates with context depth, concurrency, or both.
3. **Only if degradation appears:** A/B against `fp8_ds_mla` at matched context, holding
   every other variable fixed, to confirm KV dtype is the cause.

**The decision this informs:** if 4-bit KV is clean at the depths actually used, keep it
— it is what buys the 5.44M pool. If it degrades, the tradeoff becomes explicit: fp8 for
clean output at reduced pool, NVFP4 for maximum context.

</details>

### 5c. Prefill gap → issue #11

**Why:** the only axis where we trail. 1,512 vs 2,639 tok/s at 100K against the 2-node
recipe, while leading on every decode metric.

**Already ruled out:** node count (decode leads at TP=3), the shared-expert bug (§3),
and long-context falloff (prefill is flat ~1,000–1,075 across 25K/50K/100K).

**Already known:** prompt content moves prefill **51%** on the same server, same day —
random-word filler gives ~1,000 tok/s, repeated-sentence gives 1,512. **Their 2,639 is a
best case, not a general rate**, so the true gap is smaller than the raw numbers suggest.
Any retest must hold the prompt constant. Note their harness targets 100,000
*characters* ≈ 78K tokens — "100K" is not 100K tokens on either side.

**Leads in order:**
1. ~~**`NCCL_IB_MERGE_NICS=1`**~~ — **FALSIFIED 2026-08-24.** Measured both ways on our
   own fabric, inside the production image: **no effect at any message size**
   (0.47–0.52 GB/s busbw either way). `NCCL_DEBUG=INFO` shows NCCL **already merges both
   HCAs by default** into a 400 Gb/s virtual device and routes every channel over it.
   The upstream +64% targets a config where the merge was off. **Do not re-open.**

   **What the measurement found instead:** `GPU Direct RDMA Disabled` on every HCA —
   GB10 has no GPUDirect, so all inter-node traffic is host-staged at ~0.5 GB/s (≈2% of
   line rate). Prefill is large-message and allgather-dominated; decode is not. That
   asymmetry *is* the prefill gap, and it is architectural rather than a tuning miss.
   See [`SEQS32-AND-NCCL-FABRIC.md`](SEQS32-AND-NCCL-FABRIC.md) §3.

2. **Runtime version** — we are on 0.25.2, they are on an overlay on 0.21.1rc1. Most
   likely structural cause, most expensive to test.
3. **`MAX_NUM_BATCHED_TOKENS`** — both run 8192. Prior work here found 16384 costs 43% of
   KV for zero gain and reverted it. Retest only in a prefill-specific context, and
   expect the KV cost.

---

## 6. Where things live

| what | where |
|---|---|
| Repo | `github.com/colonel-otto/3spark-dsv4` (working copy lives under the operator's `Local LAN AI` control-room folder) |
| Open PR | [#9](https://github.com/colonel-otto/3spark-dsv4/pull/9) — this experiment, **open, not merged** |
| Open issues | [#10](https://github.com/colonel-otto/3spark-dsv4/issues/10) seqs=32 · [#11](https://github.com/colonel-otto/3spark-dsv4/issues/11) prefill · [#12](https://github.com/colonel-otto/3spark-dsv4/issues/12) KV quality |
| Latest results | `results/20260824-mtp5-1m/` — raw JSON, acceptance reps, `accept.py` |
| Live config | `~/localai/dspark-vllm-gx10/config/tp3.env` on each of the 3 nodes |
| Config backups | `tp3.env.bak-*` beside it, timestamped per experiment |
| Grafana / Prometheus | `sparkmain:3001` / `sparkmain:9090` |

### Before committing

The repo has a pre-commit hook (`.githooks/pre-commit` → `scripts/check_no_sensitive.py`)
that blocks serials and sensitive data. Run `make install-hooks` after cloning. Verify
manually with:

```bash
py scripts/check_no_sensitive.py     # or python3, depending on platform
```

Docs here contain internal IPs and hostnames, so this matters. It passed clean across
94 tracked files as of this handoff.

---

## 7. Open questions nobody has answered

- Does prefill parity with the 2-node recipe exist at all on TP=3? **Neither public repo
  publishes a TP=3 prefill number**, so there is no reference to compare against.
- Does `max_num_seqs=64` continue the trend past 32? Untested by anyone, including
  upstream.
- Is MTP=5 optimal, or would a multiple of 5 (k=10) go further? The 2-node repo reports
  k=10 boots but crashes every generation on *their* stack; untested on 0.25.2. MTP=5
  boots and runs cleanly here.
- What does abliteration cost against the stock checkpoint? Upstream states plainly that
  no censored-vs-uncensored A/B exists behind their numbers either.
