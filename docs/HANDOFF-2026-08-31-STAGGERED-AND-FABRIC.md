# Handoff — 2026-08-31: staggered gate PASSES, nvfp4 is impossible, fabric self-heals

Supersedes [HANDOFF-2026-08-30-EVENING-KSWEEP.md](HANDOFF-2026-08-30-EVENING-KSWEEP.md)
for cluster state and its "still open" list. Written for a fresh conversation.

## 1. Cluster state — READ FIRST

Unchanged from the K-sweep handoff except where noted:

- **`eugr.service` on :8100** is the DSv4 engine, 3 nodes TP=3, 1M context,
  `--kv-cache-dtype fp8`, `--max-num-seqs 16`, `--max-num-batched-tokens 8192`,
  `--gpu-memory-utilization 0.82`, prefix caching on. Serves both
  `deepseek-v4-flash-dspark-abliterated` and `deepseek-v4-flash-eugr-ab`.
  **:8000 is dead for DSv4.**
- Tuning still pinned in the unit: `EUGR_NST=5`, `EUGR_MNBT=8192`.
- **Restarted twice today** (the nvfp4 probe and its revert). Post-restart KV
  pool 2,375,397 tokens / 2.27x at 1M — the fp8 baseline. Correctness re-run
  **7/7**, virtual-TP active (heads 64→72). Gateway **4/4**.
- **LiteLLM on bigdog is already a systemd unit** — `litellm.service`, enabled,
  `MainPID` owns the running process, with a `litellm-wait-ready` readiness gate
  in `ExecStartPost`. The previous handoff's "bare nohup process" is stale. The
  work is **PR #44, open, CI green, MERGEABLE/CLEAN** — merge it.
- **New:** every node has a loopback rendezvous address —
  sparkmain `192.168.200.1`, spark1 `.2`, spark2 `.3` — plus host routes to
  each peer. `dsv4-fabric-reconcile.service` restores them every boot
  (installed on all three nodes; **enable it** — see §4).

## 2. What today settled

### The staggered ragged-context gate PASSES

First run of `benchmark_staggered_spec_acceptance.py` against the serving
config. 150 Poisson-arrival requests, 30 per tier at c=1,4,8,16,32, prompt
depths log-uniform from 1,154 to 128,275 tokens.

| cc | errors | window fails | preemptions | acceptance | TTFT median |
|---:|---:|---:|---:|---:|---:|
| 1 | 0 | 0 | 0 | 41.3% | 3.53 s |
| 4 | 0 | 0 | 0 | 40.8% | 7.79 s |
| 8 | 0 | 0 | 0 | 41.6% | 12.49 s |
| 16 | 0 | 0 | 0 | 41.3% | 61.88 s |
| 32 | 0 | 0 | 0 | 40.0% | 61.04 s |

MTP acceptance is **flat from c=1 to c=32** — twice `--max-num-seqs` — so draft
depth does not degrade under async batching. Ragged churn produced no errors and
no short completions, and the KV allocator never preempted, so the virtual-TP
zero-fill sink is stable under dynamic slot compaction.

**This is a correctness gate, not a throughput result.** The workload is
prefill-bound by construction: each tier makes a fixed 7,680 output tokens
against 442K–1.02M prompt tokens (58–132x), so the per-request decode figure
decays to 1.5 tok/s while the engine sustains ~1,460–2,300 tok/s of *prefill*.
Decode for this config remains the K-sweep's 84.3 (c=1) → 252.9 (c=8). Quoting
this run's tok/s as throughput is the aggregate-metric trap.

Bundle: `results/20260831T0601Z-staggered-spec-acceptance/`.

### `nvfp4_ds_mla` cannot close the KV gap — it is rejected for MLA

Item 2 of the old list is answered and closed. `nvfp4_ds_mla` **is** a legal
`cache_dtype` value, and a `VllmConfig` validator rejects it anyway:

> nvfp4 KV cache is not supported with MLA (Multi-head Latent Attention)
> backends. Please use a different --kv-cache-dtype (e.g. 'fp8' or 'auto') for
> MLA models such as DeepSeek.

DeepSeek-V4-Flash is MLA, so **fp8 is the floor and the KV delta vs the anemll
arm (2.36M vs 3.59M) is permanent** — same shape as the DSpark depth floor. Two
traps recorded in troubleshooting.md: probing the accepted-values `Literal` says
"supported", and the boot log prints *"Using nvfp4_ds_mla … boosts the
performance"* one second **before** the `ValidationError`. The failed boot left
**zero** leaked containers, confirming `ExecStopPost` works on this path too.

### The fabric gate was failing on a healthy fabric

`fabric_gate.sh` returned **7 failures** — every peer unreachable, every egress
off-fabric — because `configs/3spark-live.env` still named the pre-Sync
`192.168.100.2` / `192.168.101.2`. The run contradicted itself: peers
"UNREACHABLE" while the addressing and arp checks called those same peers clean
and correctly cabled. sparkmain hid the breakage because its rendezvous address
is on its loopback, which a renumber cannot touch.

Fixed by giving every node a loopback address and per-pair host routes (the
fabric is a triangle; no single interface address is reachable from both peers).
**7 failed → 21 passed, 0 failed**, mesh RTT 0.80–1.20 ms on all six pairs.

## 3. Gates

- **Fabric**: 21 passed / 0 failed with `--nccl=skip`. **Bandwidth is still
  unmeasured** — it needs the GPUs free. Run it in the next engine-down window.
- **Correctness**: 7/7 post-restart on :8100, virtual-TP confirmed.
- **Gateway**: 4/4 hops.
- **Repo**: 38 tests / 52 subtests pass.

## 4. Next steps

1. **Merge PR #44** (LiteLLM unit). It is green and clean; it was deployed and
   verified live, only the merge is outstanding.
2. **Enable the reconcile unit on all three nodes** — the files and
   `/etc/dsv4-fabric-map` are installed and were run by hand, but the unit is
   not yet enabled (that needs your approval):
   `sudo systemctl daemon-reload && sudo systemctl enable --now dsv4-fabric-reconcile.service`
   `scripts/fabric/install-fabric-reconcile.sh` does this in one call and
   `scripts/fabric/README.md` carries the exact per-node invocations.
3. **Run the NCCL bandwidth gate** in the next engine-down window — it is the
   check that actually matters and is the only one still unverified today.
4. Optional: re-run the staggered gate with a decode-shaped mixture (drop tier C
   to ~2%) if a *throughput* number under ragged load is ever wanted; today's
   run cannot provide one.

## 5. Traps added to troubleshooting.md today

`nvfp4_ds_mla` is rejected for MLA models and lies in the log before it fails;
the fabric gate fails on a healthy fabric when `FABRIC_ADDRS` is stale after a
renumber (and how to read the self-contradicting output); `quick-validate.sh`
defaulted to the dead port and **passed** on an empty body.
