# Engine A/B — 3-node TP=3, anemll-v0.25.1 vs eugr-spark-vllm-b12x

**Status: ARM 1 MEASURED (2026-08-30) — correctness gate PASSED, throughput recorded; matched same-day A/B against a live anemll engine still pending.** Results bundle: `results/20260830T194550Z-engine-ab-eugr/`. Rows: `config_id=eugr-tp3-seqs16-dspark5` in measurements.csv.
Image pulled to all three nodes 2026-08-30 at digest
`sha256:7dc02f162929943ba2e14514066ed2a04bb7e9ed3592d4eb460ebcbb1f8376bd`
(24.9GB; vLLM `0.1.dev20133+gb5f995e73.d20260823` — a main-branch build —
FlashInfer 0.6.18, torch 2.13.0+cu130, b12x vendored).

**2026-08-30 finding that rewrites the gate below: DO NOT apply our TP=3
padding patch to this image.** The image carries a native, generalized
`virtual_tp` framework (`vllm/config/virtual_tp.py` +
`model_executor/virtual_tp.py`) with explicit DeepSeek-V4 support. Verified
against our actual checkpoint (`hf-DeepSeek-V4-Flash-0731`, heads=64,
o_groups=8) at TP=3, `_build_b12x_virtual_tp_plan` produces exactly our R2
design: heads 64→72, output groups 8→9 with heads_per_group held at 8
(3 groups × 24 heads per rank), MoE intermediate 2048→2112 (same lcm(3,64)
math as our patch), vocab 129280→129408, shared-expert 2048→2304. The plan
mutates the HF config before model build and zero-fills checkpoint tails in
the weight loaders (`pad_or_narrow_weight`), so the model code sees divisible
shapes. `apply_tp3_patch.py --check --force-version` against this tree: 9/15
anchors found, 6 MISS — the MISSes are missing because the image solved them
structurally, not because they need porting. Applying our patch on top risks
double-padding.

What our patch does that theirs may not: our validated R2 analysis set
`attn_sink = -inf` on pad heads; their loaders zero-fill the sink (pad-head
outputs pass through zeroed wo_a/wo_b slabs, so the contribution *should* be
nil either way). This is exactly what the correctness gate below settles
empirically — it is the one open question.

## Scope decision (2026-08-30)

Only the **3-node TP=3** shape gets a new-engine arm. The 2-node baseline is
NOT being re-run on the new engine at this time — the 2-node numbers remain
old-engine-only and are labeled as such by the `engine` column. The question
this A/B answers is narrow: *on our production 3-node shape, does the new
engine regress or improve anything?*

## The engine is part of a measurement's identity

Every CSV row now carries an `engine` column:

- `anemll-v0.25.1` — everything through 2026-08-30: ghcr Anemll image or our
  `Dockerfile.runtime` rebuild of it. Stock vLLM v0.25.1 + b12x + 13-file
  overlay + our patch set.
- `eugr-spark-vllm-b12x` — the community image vLLM's own recipe site
  (recipes.vllm.ai) recommends for DSV4-Flash on DGX Spark. Built from vLLM
  main + FlashInfer main by eugr/spark-vllm-docker. **`:latest` moves — record
  the image digest (`docker images --digests`) in the run's results/ bundle or
  the row is not reproducible.**

Cross-engine comparison is valid ONLY cell-by-cell with everything else
matched: same harness, same prompt_shape, same concurrency, same context, same
day, healthy fabric (NCCL sanity check first — see the silent-RDMA-degradation
history). One variable: the engine.

## ⛔ Gate before ANY eugr 3-node number is recorded

**TP=3 on an unvalidated engine can SILENTLY serve nonsense** — the failure
mode that motivated our padding patch produces fluent garbage with no error.
The eugr image handles TP=3 natively via its virtual-TP plan (see Status), so
the gate is no longer "port our patch" — it is **validate their native path**:

1. Boot via the staged recipe (runbook below) and confirm in the startup log
   that the virtual-TP plan activated (heads 72, groups 9).
2. Run the correctness battery against the endpoint BEFORE any throughput
   cell: the acceptance items in `docs/patch.md`, the garble sweep, tool
   battery, and RULER-lite from `results/20260827-quality-suite-3node/`
   (unmodified scripts, checksums recorded there). Record all outputs in the
   results/ bundle. **A throughput number from an unvalidated TP=3 engine is
   not a measurement — it may be the speed of generating garbage.** The
   specific thing this checks: their zero-filled pad-head `attn_sink` vs our
   `-inf` treatment.
3. NCCL fabric sanity check first (silent-RDMA-degradation history).

## Runbook (staged 2026-08-30, dry-run validated)

Launcher: `eugr/spark-vllm-docker` cloned at `sparkmain:~/eugr-launcher`
(commit e9cf359, 2026-08-26). Our recipe lives there as
`recipes/dsv4-flash-0731-local-tp3.yaml` and is vendored in this repo at
`scripts/eugr-ab/dsv4-flash-0731-local-tp3.yaml` (keep both in sync).

Prerequisites (re-do after any node reboot — /tmp is volatile):

```bash
# on EVERY node: uniform checkpoint path for the shared -v mount.
# Docker rejects a symlink as a bind source -> hardlink farm (instant, same fs).
mkdir -p /tmp/dsv4 /tmp/hfcache && cp -al $HOME/dsv4/hf-DeepSeek-V4-Flash-0731 /tmp/dsv4/
```
Also once per head node: map each worker's wired IP to its username in
`~/.ssh/config` (the launcher SSHes workers by bare IP).

Launch (from sparkmain, inside ~/eugr-launcher; add --dry-run to preview):

```bash
# NODE0/1/2 = the WIRED (enP7s7, 10G) addresses of head, worker1, worker2, head
# first -- NOT the wifi addresses hosts.json lists. The exact command that
# booted arm 1 (all flags are load-bearing; see docs/troubleshooting.md
# "eugr launcher bring-up"):
HF_HOME=/tmp/hfcache python3 run-recipe.py dsv4-flash-0731-local-tp3   -t eugr/spark-vllm-b12x:latest   -n $NODE0,$NODE1,$NODE2   -v /tmp/dsv4:/models/dsv4host   --no-cache-dirs --gpu-memory-utilization 0.82   -e "NCCL_IB_HCA==rocep1s0f0,roceP2p1s0f0,rocep1s0f1,roceP2p1s0f1"   -e NCCL_IB_SUBNET_AWARE_ROUTING=1 -e NCCL_NET_PLUGIN=none -e NCCL_IB_MERGE_NICS=0   -e NCCL_BUFFSIZE=16777216 -e NCCL_TIMEOUT=3600
```
Run it under `nohup … > ~/eugr-ab-launchN.log 2>&1 &` -- the engine streams to
the launcher's stdout, not to `docker logs`. Never start a new attempt within a
minute of a failed one: the previous teardown is still running and the new
launcher dies silently (attempt 6 on 2026-08-30). Measured boot to
`Application startup complete`: ~8 min with cold kernel caches.

Endpoint: `http://$NODE0:8000`, served model name
`deepseek-v4-flash-eugr-ab`. Expect a long cold start (weight load + AOT
compile + CUDA-graph capture). The launcher stops workers with
`launch-cluster.sh` teardown; verify with `docker ps` on every node
afterwards (leaked-container history).

Recorded config deltas vs the anemll arm (state them in every row's notes):

- `--kv-cache-dtype fp8` (ours: `nvfp4_ds_mla`)
- speculative: dspark nst=5 probabilistic (ours: MTP K=2)
- `VLLM_USE_V2_MODEL_RUNNER=1`, AOT compile, b12x backends as per recipe
- `max_num_seqs 16` and `max_num_batched_tokens 8192` match the tp3-seqs16
  reference identity

Also pin before measuring (upstream defaults moved since v0.25.1):

- `--max-num-batched-tokens 8192` (newer default is 16384 — measured here as
  −43% KV for zero gain, see the reverted rows in measurements.csv)
- Note whether async scheduling auto-enabled (newer vLLM turns it on with
  draft models); record the speculative method and depth actually in effect —
  the eugr/recipe default is `dspark nst=7 probabilistic`, ours is MTP K=2, and
  those are different measurements.
- Record `prefix_cache_retention_interval` / prefix-cache state, kv-cache-dtype
  and `gpu_memory_utilization` as launched.

## The cells

Old-engine reference values are QUERIED from benchmarks/summary.csv (they are
restated here for convenience; summary.csv wins on any discrepancy). Fill the
new-engine column by appending rows to measurements.csv with
`engine=eugr-spark-vllm-b12x` and the SAME config/harness/prompt columns —
the comparison table then falls out of the data instead of being hand-kept.

| cell | harness | prompt_shape | c | anemll-v0.25.1 | eugr-spark-vllm-b12x |
|---|---|---|---|---:|---:|
| single-stream decode (256-tok prompt) | bench-miaai | synthetic-numbered-words | 1 | 80.4 (tp3-seqs16) | **82.1** (warm; cold read 65.4 under JIT compiles) |
| aggregate c=4 | bench-miaai | synthetic-numbered-words | 4 | 115.2 | **162.7** (+41%) |
| aggregate c=8 | bench-miaai | synthetic-numbered-words | 8 | 143.6 | **171.7** (+20%) |
| peak useful aggregate (seqs cap) | bench-miaai | synthetic-numbered-words | 16 | 161.0 | 133.9 (−17%; c=16 scheduling cliff, TTFT 7s — see bundle README) |
| decode at 131,072-token context | bench-miaai | synthetic-numbered-words | 1 | 83.5 | — |
| prompt-effect: code-brief | ours-bench.py | code-brief | 1 | 81.8 | — |
| prompt-effect: dense-prose | ours-bench.py | dense-prose | 1 | 49.4 | — |
| deep concurrency 4×~200K (usability) | deepconc.py | synthetic-numbered-words | 4 | 0.9 (unusable) | — |
| KV cache tokens (capacity, prompt-independent) | n/a | n/a | n/a | 3,588,422 | 2,415,674 (kv fp8 vs nvfp4_ds_mla delta) |

Decode rates depend on the speculative path (the prompt-effect pair exists
because MTP acceptance moves decode 1.65x). If the eugr arm runs a different
speculative config than MTP K=2, the decode cells measure *engine+speculator*,
not engine — still useful, but say so in the row notes.

## Where results go

- Raw evidence: a `results/<UTC>-engine-ab-eugr/` bundle (digest, launch
  command, engine startup log, harness outputs, NCCL sanity check).
- Rows: append to `benchmarks/measurements.csv` with the new engine label;
  regenerate summary.csv (`python3 scripts/generate_summary.py`); schema test
  enforces the engine whitelist (`tests/test_benchmark_schemas.py`).

## Arm 1 verdict and next steps (2026-08-30)

Correctness: quick gate 7/7, tool battery 6/7 (`forced_choice` = valid JSON, API semantics), deep-context 8/8,
garble ALL CLEAN, RULER-lite 16/16. The native virtual-TP path is correct at TP=3; our patch stays off.

Throughput: single-stream parity; +20–41% aggregate at c=4–8; −17% at the c=16 cap because of a scheduling
cliff (TTFT 7.0s vs 1.9s at c=8) that the engine itself attributes to speculative draft slots
(`max_num_scheduled_tokens is set to 8128 … decrease num_speculative_tokens or max_num_seqs`).

Next boot, in this order (one variable each):
1. Persist kernel caches. **STAGED AND DRY-RUN VALIDATED 2026-08-30** as
   `scripts/eugr-ab/eugr-boot.sh` — but NOT the way this list originally said. Two
   corrections, both in docs/troubleshooting.md:
   - `--no-cache-dirs` must **stay on**. The launcher mounts caches by default and that
     default is `$HOME`-relative, which breaks on the workers (homes are
     `/home/sparkmain`|`spark1`|`spark2`). Keep the flag to suppress the broken mounts and
     supply uniform absolute paths with `-v`, which the launcher forwards unchanged.
   - `/opt/eugrcache-*`, not `/tmp/eugrcache-*` — systemd-tmpfiles wipes `/tmp` on reboot,
     which would silently re-cold every kernel cache after any node restart.
   Dirs are pre-created on all three nodes; the boot script re-creates them as a
   precondition. Verify after the boot that `grep -c 'cute.compile.*disk-cache-miss'`
   stops growing across boots.
2. The K sweep: nst ∈ {2,3,5,7} × c ∈ {1,4,8,16}, plus `max_num_batched_tokens 16384`
   at nst=5 as the alternative lever on the same cliff. nst=2 first (it matches our MTP
   K=2, removing the speculator delta from the cross-engine comparison). Driver:
   `scripts/eugr-ab/eugr-sweep.sh`, which warms the caches, asserts the JIT miss counter
   has frozen before recording, then runs median-of-5 per cell into `rows.tsv`.
   Harness `scripts/eugr-ab/bench-miaai.py` is the byte-identical arm-1 harness (it lived
   only in volatile `/tmp`; now vendored).
   Note on framing: depth-5 drafts were accepted ~4.7-4.9/5, so deep drafts are NOT being
   wasted — the c=16 penalty is scheduler budget (`max_num_scheduled_tokens=8128`), not
   acceptance. Expect low K to win at concurrency and high K single-stream; the gateway
   serves the winner.
3. Restore the LAN gateway route on the same boot — folded into `eugr-boot.sh`:
   `--port 8100` plus BOTH served names
   (`deepseek-v4-flash-dspark-abliterated deepseek-v4-flash-eugr-ab`), so bigdog's LiteLLM
   (:4000) and the manifest service (:8771) resolve with no client changes. The generated
   recipe carries both names; the boot script's dry-run gate asserts both are present.
   Verify end-to-end through bigdog:4000 after the boot, not just on :8100 locally.
4. Make eugr a systemd service wrapping this boot command, with `ExecStopPost` teardown
   and a `docker ps` check on all three nodes (leaked-container history).
5. Only then consider `--kv-cache-dtype nvfp4_ds_mla` if this build supports it, to remove the KV delta.
6. The matched A/B proper: same day, anemll engine live, same harness, both arms — per
   feedback "measure our own A/B first".
