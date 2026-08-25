# Upper mesh addressed: the four-HCA path, re-tested

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

## Status: promising, NOT yet adopted

What this establishes:

- ✅ The +56% figure was **not** an artifact — it is now **~2.0x**, on a path that is
  properly addressed, routed, persisted, and gate-clean.
- ✅ The **3-rank** number moves too (2.85 → 5.80 GB/s), which matters most: the 3-rank
  collective is what bounds TP=3.
- ✅ It narrows the gap against published 3-Spark rings ([#11](../../issues/11)).

What it does **not** establish:

- ❌ **No engine run.** The gate measures NCCL collectives with the engine stopped. The
  previous attempt also passed initialisation and *then* wedged under real load —
  **init success is not health.** This is exactly the failure class the postmortem names.
- ❌ **No tok/s.** Fabric bandwidth is not throughput. Decode is not obviously
  bandwidth-bound at cc=1, so a 2x fabric gain may deliver little.
- ❌ **No sustained-load soak.** `IBV_WC_RETRY_EXC_ERR` appeared under traffic, not at init.

## Next step

Start the engine with all four HCAs, then in order:

1. `make gate CONFIG=configs/3spark-live.env` — engine up, checks `rdma:*` counters
   (they `skip` here because the engine was stopped; **that is the check that would have
   caught the previous wedge**).
2. Correctness (17×23 → 391) — the padding patch means a broken fabric can serve
   *fluent nonsense*.
3. Warm shapes, then decode at cc=1/4/8/16 against
   [`../20260825-decode-2v3/`](../20260825-decode-2v3).
4. Soak under sustained load, watching for `IBV_WC_*_ERR`.

Until steps 1–4 pass, `config/tp3.env.example` keeps `NCCL_IB_HCA=rocep1s0f0,rocep1s0f1`.

## Files

| File | Contents |
|---|---|
| `gate.json` | 2-HCA baseline, 26/26 pass |
| `gate-four-hca.json` | 4-HCA run, 26/26 pass |
