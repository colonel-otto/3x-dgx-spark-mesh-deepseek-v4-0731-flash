# Raw run bundles

**Every directory here is frozen.** It records one experiment at the configuration and
fabric state it was measured under. Do not edit one in place — supersede it with a new
dated directory and mark the old one superseded in this table.

Directories are named `YYYYMMDD[THHMMSSZ]-<subject>`.

| Fabric | Meaning |
|---|---|
| 🟢 healthy | Measured after the 2026-08-25 fabric fix. Trustworthy |
| 🔴 degraded | One node at ~15% collective bandwidth. **See [`../docs/DEGRADED-DATA-CATALOGUE.md`](../docs/DEGRADED-DATA-CATALOGUE.md)** |
| ⚪ n/a | Not a performance measurement |

---

## 2026-08-25 — healthy fabric

| Directory | Fabric | What it holds |
|---|---|---|
| [`20260825-fabric-fix/`](20260825-fabric-fix) | ⚪→🟢 | The fix itself: pre/post-reboot prefill and decode. **The before/after pair** |
| [`20260825-prefill-2v3/`](20260825-prefill-2v3) | 🟢 | Prefill, 2 vs 3 nodes. Result: **parity (±2%)** |
| [`20260825-decode-2v3/`](20260825-decode-2v3) | 🟢 | Decode, 2 vs 3 nodes at cc=1/4/8/16. **The headline result** |
| [`20260825-deep-concurrency/`](20260825-deep-concurrency) | 🟢 | 4×200K re-run for [#15](../../issues/15). Includes `deepconc.py` and the gate output |
| [`20260825-upper-mesh/`](20260825-upper-mesh) | 🟢 | **Four-HCA fabric at 2.0x**, 26/26 gate-clean. Engine-validation pending — [#11](../../issues/11) |

## 2026-08-24 — degraded fabric

Conclusions may hold; numbers should not be quoted without a re-run.

| Directory | Fabric | What it holds | Status |
|---|---|---|---|
| [`20260824-mtp5-1m/`](20260824-mtp5-1m) | 🔴 | MTP=5 vs 4, 1M context | ✅ Conclusion survives — matched arms, same handicap both sides |
| [`20260824-prefill/`](20260824-prefill) | 🔴 | 45 files: the long prefill investigation | ⚠️ The ~2x "gap" chased here was **one degraded node** |
| [`20260824-seqs32-nccl/`](20260824-seqs32-nccl) | 🔴 | `seqs=32` + NCCL sweeps. Includes `agbench.py` | ⚠️ Rejected against a budget **6.6x too small** — [#10](../../issues/10) |
| [`20260824-kv-quality/`](20260824-kv-quality) | 🔴 | NVFP4 KV quality to 464K. Includes `kvquality.py` | ⚠️ Single-arm, no comparison — [#16](../../issues/16) |

## 2026-08-21 — degraded fabric, original sharding experiments

| Directory | Fabric | What it holds | Status |
|---|---|---|---|
| [`20260821T001024Z-2spark-baseline/`](20260821T001024Z-2spark-baseline) | 🔴 | The frozen 2-node reference, 48.23 tok/s | 🧊 Historical |
| [`20260821T031300Z-3spark-ep3/`](20260821T031300Z-3spark-ep3) | 🔴 | EP=3: per-rank configs + routes | ❌ 2.5x slower; kernel finding stands |
| [`20260821T133000Z-3spark-pp3/`](20260821T133000Z-3spark-pp3) | 🔴 | PP=3 across several shapes | ❌ Hard block, **not** a perf number — survives |
| [`20260821T133000Z-3spark-tp3/`](20260821T133000Z-3spark-tp3) | 🔴 | TP=3 rank config + mesh setup | 🧊 Historical |
| [`20260821T142000Z-3spark-tp3-upstream-harness/`](20260821T142000Z-3spark-tp3-upstream-harness) | 🔴 | Upstream's harness, unmodified. Note `warmup-discarded.json` | 🧊 Historical |

---

## Adding a run

1. **Gate first:** `make gate-full CONFIG=configs/3spark-live.env`, engine stopped. Save
   the gate JSON *into the run directory* — it is the provenance for everything else.
2. **Warm your shapes**, then discard any run whose log contains
   `JIT compilation during inference`.
3. Create `YYYYMMDD-<subject>/` with a `README.md` stating **date, fabric state, config,
   image, and the question the run answers**.
4. Keep the harness script alongside its output. A result whose harness does not exist as
   a file cannot be checked — that has bitten us.
5. Add a row above, and append machine-readable rows to
   [`../benchmarks/measurements.csv`](../benchmarks/measurements.csv).
6. Never publish only the best run. Retain the controls that failed.
