# Parity reached — a silent 6.8x fabric degradation on spark1 — 2026-08-25

**Prefill is now at 95–99% of the upstream reference. Decode improved 31–59%. No
configuration change was involved: one node's RDMA fabric had silently degraded, and a
reboot cleared it.**

---

## 1. Result

Upstream's own `benchmark_prefill.py`, unmodified, server-side timer, cold prompts
(fresh seed so prefix caching cannot hit), client agreeing within 1%:

| input tokens | before fix | **after fix** | gain | anemll reference | **% of reference** |
|---:|---:|---:|---:|---:|---:|
| 1,024 | 1,063 | **2,022.6** | +90% | 2,033.0 | **99.5%** |
| 8,192 | 1,075 | **2,069.5** | +93% | 2,184.2 | **94.8%** |
| 32,768 | 1,034 | **2,094.9** | +103% | 2,176.1 | **96.3%** |

Decode, warm, idle engine, 3 reps:

| | before | **after** | gain |
|---|---:|---:|---:|
| cc=1 median | 80.4 | **85.6** | +6% |
| cc=16 median | 374.2 | **491.0** | +31% |
| cc=16 peak | 374.2 | **593.1** | +59% |

The whole ~2x prefill gap this repo has been chasing was one degraded node.

## 2. What was wrong

Every NCCL collective involving spark1 ran at ~15% of the healthy pair's bandwidth.
Pairwise 64 MiB allgather busbw:

| pair | busbw | |
|---|---:|---|
| sparkmain <-> spark2 | **4.60–4.64 GB/s** | healthy (reproduced twice) |
| sparkmain <-> spark1 (original cable) | 0.69 GB/s | degraded |
| sparkmain <-> spark1 (**alternate cable**) | 0.68 GB/s | degraded |
| spark1 <-> spark2 | 0.71 GB/s | degraded |
| all three ranks | 0.49 GB/s | paced by the worst link |

Testing the **alternate cable** matters: each node has four RDMA ports and only two are
used. Bringing up the unused `roceP2p1s0f*` pair gave the same 0.68 GB/s, which proved
the fault was **the node, not a cable**. That saved a pointless hardware swap.

### There were no error indicators at all

| check | spark1 | healthy nodes |
|---|---|---|
| port state (all 4) | ACTIVE / LinkUp | same |
| port speed | 200,000 Mb/s | same |
| `link_downed`, `rcv_errors`, `symbol_error` | 0 | 0 |
| NIC firmware | 28.45.4028 | 28.45.4028 |
| PCIe link | 32 GT/s x4 | 32 GT/s x4 |
| NCCL transport chosen | `NET/IB/2` (merged 400 Gb/s) | same |
| GPU clock | 2411 MHz | 2411 MHz |

**Nothing in `ibstat`, `nvidia-smi`, `lspci`, or the NCCL logs showed a problem.** Only a
direct NCCL collective benchmark exposed it.

### The fix

`sudo reboot` on spark1. Bandwidth went 0.69 -> **4.78 GB/s**, matching the healthy pair.

Root cause of the degradation itself is **not known**. The node had been up a long time.
If it recurs, that is worth investigating properly.

## 3. This invalidates prior measurements — see issue #14

Every multi-node number recorded before 2026-08-25 was taken with one of three nodes at
~15% collective bandwidth. **Tracked in
[issue #14](https://github.com/colonel-otto/3spark-dsv4/issues/14)**; treat the following
as provisional (*) until re-run:

- **Decode baselines** (374.2 cc=16, ~80 cc=1) — these anchor most other conclusions.
- **`MAX_NUM_SEQS=32` rejection** — it died on an `_ALLGATHER_BASE` timeout with KV at
  2.8%. A degraded link is a *plausible cause of that crash*. **seqs=32 may well be
  viable on a healthy fabric** and deserves a clean retest.
- **2-node vs 3-node** — spark1 was in the 3-node arm, so the comparison was unfair to
  three nodes.
- **EP=3 / PP=3** — communication-heavy, disproportionately affected.
- **`GPU_MEMORY_UTILIZATION` 0.80 vs 0.85** — the +14% was measured on the bad fabric.
- **MTP=4 vs 5** — acceptance counters are compute-local and probably fine; aggregate
  throughput numbers are suspect.

## 4. The encouraging read

**The degraded-fabric numbers were still good enough to lead the 2-node reference on
every decode metric.** We were beating published results while running one node at 15%
of its collective bandwidth.

Now that the fabric is healthy, the ceiling is higher than anything measured here:

- Every previously "settled" tuning conclusion was reached under a communication
  handicap. Parameters rejected because they cost communication — **`MAX_NUM_SEQS=32`
  most obviously, but also EP=3 and PP=3** — were rejected on a fabric that made
  communication ~7x more expensive than it should have been.
- The 0.49 GB/s 3-rank figure that framed the whole "GB10 has no GPUDirect, ~0.5 GB/s is
  the ceiling" analysis was **measured on the degraded fabric**. The healthy pairwise
  number is 4.6 GB/s. The real 3-rank ceiling has not yet been measured.
- Decode already gained +59% at peak from the fix alone, with no tuning.

**Re-running the tuning matrix on the healthy fabric is likely to find better settings
than we have today**, not merely re-confirm the old ones.

## 5. Persistence — fixed, was a live outage risk

spark1's reboot lost runtime-only network state and **the cluster would not start**:

- `192.168.200.1` (the master address) existed only on sparkmain's loopback at runtime.
  Had sparkmain rebooted, the cluster would have been unrecoverable without knowing this.
- spark1 lost its host route to `192.168.200.1` and fell back to routing via **WiFi**,
  so workers could not reach the master (600 s TCPStore timeout).
- spark2 had no route to spark1's `192.168.100.2`, so Gloo's `connectFullMesh` failed.

All now persisted via NetworkManager:

```bash
# sparkmain
sudo nmcli con mod lo +ipv4.addresses '192.168.200.1/32'
# spark1
sudo nmcli con mod dac-link        +ipv4.routes '192.168.200.1/32 192.168.100.1'
sudo nmcli con mod dac-link-spark2 +ipv4.routes '192.168.101.2/32 192.168.102.2'
# spark2
sudo nmcli con mod dac-link-spark1 +ipv4.routes '192.168.100.2/32 192.168.102.1'
```

Gloo needs a **full mesh**: every rank must reach every other rank's advertised
`VLLM_HOST_IP`. Verify all 6 directions before starting.

## 6. Method note

Add a pairwise fabric check to the pre-benchmark routine. A healthy GB10 pair reports
**~4.6 GB/s** busbw at 64 MiB via `results/20260824-seqs32-nccl/agbench.py`; anything
near 0.7 GB/s means a degraded node and every number taken is worthless.

TCP throughput is **not** a valid check — it showed the degraded link at 858 MB/s vs
1,019 for a healthy one (1.19x), hiding a 6.8x RDMA deficit. TCP does not exercise the
RDMA verbs path. Use the NCCL collective, which is what vLLM actually uses.

## 7. Raw data

`results/20260825-fabric-fix/` — post-fix prefill with fresh seed (the clean parity
measurement), the cache-contaminated confirmation run kept as a caution, and decode runs.
