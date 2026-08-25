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

### What is still outstanding

- ⏳ **Sustained-load soak.** `IBV_WC_RETRY_EXC_ERR` appeared under traffic, not at init.
  An immediate inference check is not a soak.
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
