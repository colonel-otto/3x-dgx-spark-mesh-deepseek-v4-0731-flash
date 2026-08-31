# Handoff — 2026-08-30 evening: K sweep done, eugr is the service, gateway live

Supersedes [HANDOFF-2026-08-30-ENGINE-AB.md](HANDOFF-2026-08-30-ENGINE-AB.md) for
cluster state and for its "next steps" list. Written for a fresh conversation.

## 1. Cluster state — READ FIRST

- **`eugr.service` is the DSv4 engine**: enabled + active on sparkmain, 3 nodes
  TP=3, 1M context, kv fp8. **Port 8100**, serving BOTH
  `deepseek-v4-flash-dspark-abliterated` (gateway route) and
  `deepseek-v4-flash-eugr-ab` (A/B row identity). **:8000 is dead for DSv4** —
  any client still pointed there must move.
- Tuning is pinned in the unit: `EUGR_NST=5`, `EUGR_MNBT=8192` (the sweep winner).
  To re-tune: edit those two Environment lines, `daemon-reload`, `restart`. The
  unit blocks until :8100 actually serves, and `ExecStopPost` tears down all three
  nodes on every exit path (proven: the nst=2 failure left zero leaked containers).
- **Persistent kernel caches** at `/opt/eugrcache-{vllm,flashinfer,triton,tilelang}`,
  ~54MB and identical on all three nodes. Keep `--no-cache-dirs` ON (it suppresses
  the launcher's broken `$HOME`-relative mounts) and pass uniform absolute paths
  with `-v`. Boot is now ~4.3 min, was ~8.
- `dsv4.service` (anemll) remains disabled/inactive. Do not start it while eugr
  holds the GPUs — `Conflicts=dsv4.service` is declared, but don't rely on it.
- Node addresses live in `sparkmain:~/.eugr-nodes` (gitignored); repo has placeholders.
- **LAN gateway is LIVE**: bigdog LiteLLM :4000 → sparkmain :8100, verified with a
  real completion round-trip. `scripts/eugr-ab/verify-gateway.sh` checks all four
  hops and currently passes 4/4.

## 2. What the sweep settled

| c | nst=5 mnbt=8192 | nst=7 mnbt=8192 | nst=5 mnbt=16384 |
|---|---:|---:|---:|
| 1 (decode) | **84.3** | 79.5 | 83.5 |
| 4 | 152.8 | 151.2 | **165.0** |
| 8 | **252.9** | 208.8 | 241.8 |
| 16 | 198.8 | 197.2 | **214.3** |
| KV tokens | **2,357,009** | 2,405,070 | 1,165,679 |
| max conc @1M | **2.25x** | 2.29x | 1.11x |

- **Depth is SETTLED at nst=5.** The legal range is only {5,7}: the checkpoint sets
  `dspark_block_size: 5` and nst<5 is *rejected* ("produce incorrect output").
  nst=7 never wins a cell. The anemll expectation "high K wins single-stream" does
  NOT transfer.
- **mnbt=16384 rejected**: +8% on two cells for **−50.5% KV cache** (max concurrency
  at 1M ctx 2.25x → 1.11x). Silences the engine's own warning; costs more than the
  disease. Same trap as anemll, now confirmed on fp8 KV too.
- *KV figures corrected 2026-08-31.* An earlier revision of this table listed
  2,415,674 for both nst columns. That number is **arm 1's** boot
  (`20260830T194550Z-engine-ab-eugr`), not this sweep's; each arm here reports
  what its OWN engine log recorded. The conclusion is unchanged (−50.5% vs −52%),
  and nst=5/nst=7 KV differ slightly (2.0%) rather than being identical.
- **Arm-1's "c=16 scheduling cliff" is RETRACTED** — mostly JIT contamination.
  Cache persistence alone gave +47% at c=8 and +48% at c=16, TTFT 7000→1755ms.
  Every arm-1 `--no-cache-dirs` row is a LOWER BOUND, not engine capability.

Evidence: `results/20260830T2245Z-eugr-ksweep/` (three rows.tsv, per-cell logs,
JIT counters, engine configs, fabric gate JSON).

## 3. Gates

- **Fabric**: 30 passed / 0 failed INCLUDING the NCCL bandwidth check
  (9.08–9.27 GB/s per pair) — only measurable with the engine down, so it was run
  in that window. Artifact in the bundle.
- **Correctness**: 7/7 at nst=5 on :8100, virtual-TP plan confirmed active
  (heads 64→72, groups 8→9). Our `apply_tp3_patch.py` stays OFF.

## 4. Next steps

1. ~~Remaining A/B cells~~ **DONE 2026-08-31** —
   `results/20260831T0030Z-eugr-remaining-cells/`. Headlines: the **prompt effect
   is larger on this engine (1.95x vs anemll's 1.65x)**; 131K cold decode is
   42.3 vs 83.5 but the configs differ (`max_model_len` 460800 vs 1048576) and
   prefill is **2.6x faster**; 4×200K is 1.4 tok/s with TTFT 227s — completes,
   still unusable. Two caveats travel with these numbers: the dense-prose prompt
   is a **reconstruction** (`ours-bench.py` was never committed), so only the
   within-engine ratio is sound; and the 131K row is not a matched config.
   Rows appended to `benchmarks/measurements.csv`; summary regenerated.
2. ~~Re-baseline the cross-engine A/B table~~ **DONE 2026-08-31** — the four
   concurrency cells in `ENGINE-AB-3NODE.md` now read from the nst=5/mnbt=8192
   sweep rows instead of the contaminated arm-1 column. **This changed the
   headline**: c=16 flipped from −17% (reported as a "scheduling cliff") to
   **+24%**, and c=8 went from +20% to **+76%**. On warm caches the new engine
   wins every concurrency cell. The PERMANENT speculator caveat (anemll MTP K=2
   vs eugr DSpark K≥5, parity impossible) is now stated above the table.
3. ~~Append rows to `benchmarks/measurements.csv`~~ **DONE 2026-08-31** — all of
   it. The 4 remaining-cells rows, plus all **12 K-sweep points** under three new
   config ids (`eugr-tp3-seqs16-dspark5-mnbt8192` — the winner and what the
   service serves, `…-dspark7-mnbt8192`, `…-dspark5-mnbt16384` marked
   `reverted=true`). The remaining-cells rows were re-homed from the transient id
   `eugr-tp3-nst5-mnbt8192` onto the winner's id: they measured the same served
   tuning, so they are the same configuration. The arm-1 rows
   (`eugr-tp3-seqs16-dspark5`) are kept and marked superseded. summary.csv
   regenerated; all 7 test scripts pass.
4. **Make LiteLLM a systemd unit on bigdog.** It runs as a bare nohup process, so
   it does not survive a reboot and a config edit needs a manual restart.
5. Optional: `--kv-cache-dtype nvfp4_ds_mla` on this build to remove the KV delta
   vs anemll, if supported.

## 5. Traps added to troubleshooting.md today

DSpark depth floor (checkpoint-set, not a flag); `--no-cache-dirs` numbers are
lower bounds and a monotonic decay across identical trials is a JIT signature;
mnbt=16384 KV trap on this engine too; `models-manifest-serve` has no `/v1/models`
(it serves unknown paths as static files, so a wrong path returns an HTML directory
listing with HTTP 200 and looks broken while healthy — use `opencode.gateway.json`);
launcher cache mounts are `$HOME`-relative and break on workers; sweep parameters
belong in a generated recipe, not `--` passthrough.
