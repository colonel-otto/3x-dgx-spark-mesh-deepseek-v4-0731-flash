# Upper mesh addressed: the four-HCA path, re-tested

**Status:** `CURRENT` within the provenance caveats in [`../index.yaml`](../index.yaml).

**Date:** 2026-08-25T23:12–23:15Z · **Fabric:** 🟢 healthy · **Engine:** stopped

Re-tests the `roceP2p` HCA pair **after giving it the IPv4 addressing and routing it
previously lacked** — the precondition the earlier rollback explicitly named.

## Result

Both gate runs pass **26/26 checks**, minutes apart, same cluster, engine stopped. The
only variable is `NCCL_IB_HCA`.

| busbw @64MiB | 2 HCAs (`gate.json`) | 4 HCAs (`gate-four-hca.json`) | gain |
|---|---:|---:|---:|
| sparkmain ↔ spark1 | 4.63 GB/s | **9.51** | 2.05x |
| sparkmain ↔ spark2 | 4.73 | **9.70** | 2.05x |
| spark1 ↔ spark2 | 4.60 | **9.15** | 1.99x |
| **all three ranks** | 2.85 | **5.80** | **2.04x** |

Transport confirms both devices are in use:

| | transport |
|---|---|
| 2 HCAs | `via NET/IB/2` |
| 4 HCAs | `via NET/IB/4` **and** `via NET/IB/5` |

## Why this is not the earlier falsified experiment

The [previous attempt](../../docs/DECISIONS.md) wedged the cluster and was rolled back.
Its failure mode was **`IBV_WC_RETRY_EXC_ERR` with both GIDs `fe80::`** — RDMA over an
unaddressed, unrouted path. The `roceP2p` pair had only link-local IPv6.

It now has full IPv4 on dedicated `/30` subnets, and the gate verifies them:

| node | upper-mesh addresses |
|---|---|
| sparkmain | `enP2p1s0f0np0=192.168.110.1/30` · `enP2p1s0f1np1=192.168.111.1/30` |
| spark1 | `enP2p1s0f0np0=192.168.112.1/30` · `enP2p1s0f1np1=192.168.110.2/30` |
| spark2 | `enP2p1s0f0np0=192.168.111.2/30` · `enP2p1s0f1np1=192.168.112.2/30` |

`persist:*` passes on all three, so these survive a reboot — the tmpfs trap that bit us
before is not in play here.

> **Note on the interface name.** These are `enP2p1s0f*` (netdev), not `roceP2p1s0f*`
> (RDMA device). The gate's subnet and persistence checks originally matched `roceP2p`
> against `ip addr` output and therefore **silently matched nothing** — the upper mesh was
> invisible to the gate. Fixed in the same commit as this directory.

## Status: ADOPTED for serving; soak still outstanding

Applied 2026-08-25T23:22Z. All six uppercase controllers given persistent IPv4 `/30`
addressing, every link verified with jumbo frames and IPv4 RoCEv2 GIDs, conflicting legacy
autoconnect profiles removed (backups in
`/etc/netplan/dsv4-backup-pre-upper-mesh-20260825/`). All four HCAs enabled on every rank;
workers restarted first, then the head.

### Live gate with the engine RUNNING — `live-gate.json`

**21 passed, 0 failed**, bandwidth skipped (engine up, as designed).

The decisive line is `rdma:*`:

| check | result |
|---|---|
| `rdma:sparkmain` / `rdma:spark1` / `rdma:spark2` | **pass, 0 errors** |

**This is the check that skipped in the stopped-engine runs, and the one that would have
caught the previous wedge.** The earlier attempt failed here with
`IBV_WC_RETRY_EXC_ERR` and both GIDs `fe80::`. No GID changes, no NCCL warnings, and no
RDMA completion errors appeared this time.

API health and inference both passed. KV capacity after restart: **4,502,448 tokens**.

### What is confirmed

- ✅ Bandwidth doubles and is **real under a live engine**, not just a stopped-engine gate.
- ✅ Zero RDMA completion errors — the specific failure mode of the earlier rollback.
- ✅ Addressing is persistent, so a reboot will not silently revert it.
- ✅ Follows NVIDIA's guidance that each logical controller gets a unique address/subnet.

### Sustained soak — PASS (2026-08-26T00:16-00:37Z)

The outstanding item is closed. 20.5 min, 8 concurrent streams, mixed 1K/4K/16K shapes.

| | |
|---|---|
| requests | **408 / 408 successful, 0 failures** |
| latency | p50 21.86s · p95 45.73s · p99 58.62s |
| volume | 2,347,035 prompt + 16,071 completion tokens |
| **RDMA counter deltas** | **0 on all 132 counters, all three hosts** |
| log patterns | **0 hits** for `IBV_WC` / `RETRY_EXC` / `GID table changed` / `NET/Socket` / NCCL warn/error |
| container restarts | none — all up across the whole window |
| correctness | **391** ✓ |
| gate (engine up) | **24 pass / 0 fail / 1 skip**, `rdma:*` clean on all three |

**The failure mode from the earlier attempt did not reproduce.** That attempt wedged under
traffic *after* passing init; this one carried 2.3M prompt tokens with zero RDMA events.

#### The counter trap this surfaced

The `roceP2p` pair shows **nonzero absolute** error counters on every host, identically:
`local_ack_timeout_err=192, req_cqe_error=64, req_cqe_flush_error=32, resp_cqe_error=128,
resp_cqe_flush_error=96`. The lower `rocep1s0f*` pair reads zero.

**These are pre-existing residue, not a live fault.** Verified frozen twice — across a 45 s
idle sample, and again mid-soak under active load. They are cumulative since boot, left by
the earlier failed enable, and nothing clears them short of a reboot.

**Judge these counters by DELTA, never by absolute value.** An absolute-value check would
false-positive permanently. The gate is already immune: `rdma:*` reads the engine log
(live events), not sysfs (lifetime totals).

### What is still outstanding

- ✅ ~~Sustained-load soak~~ — **done, PASS.** See above.
- ⏳ **No tok/s yet.** Fabric bandwidth is not throughput. Prefill measured at *parity*
  between 2 and 3 nodes, which is evidence prefill is not fabric-bound — so a 2x fabric
  gain may deliver little. Decode at cc=1/4/8/16 against
  [`../20260825-decode-2v3/`](../20260825-decode-2v3) is the test that matters, and the
  interesting question is **whether the cc=16 crossover moves.**

## Files

| File | Contents |
|---|---|
| `gate.json` | 2-HCA baseline, 26/26 pass |
| `gate-four-hca.json` | 4-HCA run, 26/26 pass |
| `live-gate.json` | **Engine running**, 21/21 pass, `rdma:*` clean on all three ranks |
