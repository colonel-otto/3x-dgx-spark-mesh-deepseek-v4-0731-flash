# The matched test: reproduce NVIDIA's recipe exactly

**We cannot yet call 5.80 vs 20.84 GB/s a controlled comparison.** Four variables differ
between our measurement and the only well-documented public 3-Spark result. This page
fixes them one at a time.

Read [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md) first for what the gap is and is
not.

---

## The evidential situation, stated honestly

The 18-21 GB/s figure rests on **one** well-documented public 3-Spark result
([forum 365160](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160)):

| | busbw |
|---|---:|
| their **first** measurement | **2.86 GB/s** <- almost exactly our original number |
| after fixing the OOB/bootstrap path | 18.64 @32 MiB, 20.84 @16 GiB |
| NVIDIA staff expectation | ~24 GB/s |

**That they started where we are is the most useful fact available.** Their fix was not a
fabric change -- it was how the job bootstraps.

## Four variables that differ

### 1. Bootstrap / control-plane topology <- strongest suspect

**Verified in our code.** `scripts/fabric_gate.sh` bootstraps over the **fabric**:

```bash
master=$(fabric_addr_for "${group[0]}")
-e INIT_METHOD=tcp://$master:29555
```

and the serving config pins each rank to a *different* QSFP-facing interface:

```bash
NCCL_SOCKET_IFNAME=enp1s0f0np0     # per-rank, differs per node
MASTER_ADDR=192.168.200.1          # on the head's LOOPBACK
```

NVIDIA's [`launch.sh`](https://github.com/NVIDIA/dgx-spark-playbooks/blob/main/nvidia/nccl/assets/launch.sh)
uses **management IPs and one common interface on every node** -- the playbook sets
`UCX_NET_DEVICES`, `NCCL_SOCKET_IFNAME` and `OMPI_MCA_btl_tcp_if_include` all to `enP7s7`.
Wi-Fi is explicitly supported for this path, which tells you it is a *control* plane, not
a data plane.

### Do the Sparks need Ethernet cables? **No.**

This is worth stating plainly, because "management interface" sounds like it implies a
cable. It does not.

**Measured 2026-08-25 on all three nodes:**

| node | `enP7s7` (Ethernet) | `wlP9s9` (Wi-Fi) |
|---|---|---|
| sparkmain | **down** | 192.168.1.223 |
| spark1 | **down** | 192.168.1.50 |
| spark2 | **down** | 192.168.1.27 |

**Not one Spark is on Ethernet.** All three reach the LAN over Wi-Fi, and all six
node-to-node paths on 192.168.1.x are reachable (3-135 ms RTT). The shared control network
we need **already exists** -- no cabling required.

The ConnectX-7 ring stays exactly as it is. It carries **all** the data. The only thing
that would move is the few kilobytes of rendezvous traffic NCCL uses to agree on a
topology before any payload flows -- and NVIDIA's playbook **explicitly supports Wi-Fi**
for it, which is the clearest possible signal that it is a control plane, not a data path.

> **Terminology note:** NVIDIA's playbook names `enP7s7` because that is the Ethernet port
> on their reference setup. What matters is *one interface, common to every node, that is
> not the fabric* -- `wlP9s9` satisfies that here. Substitute it throughout.

Bootstrap traffic is tiny; RDMA still carries the payload. Putting rendezvous on a
rank-specific fabric interface -- with the master address on loopback -- is a plausible way
to get a wrong or asymmetric topology decision, and it is exactly the class of thing the
forum reporter fixed to go from 2.86 -> 18.64.

> **Note:** the gate does **not** set `NCCL_SOCKET_IFNAME` at all (verified: 0 occurrences).
> It only sets `INIT_METHOD` to a fabric address. The serving config *does* pin it. So the
> two paths differ from each other as well -- worth keeping straight when comparing.

### 2. NIC merging was never tested correctly

The forum's working ring ran **`NCCL_IB_MERGE_NICS=0`** -- explicitly off.

**We have never set this variable in either path** (verified: 0 occurrences in
`fabric_gate.sh`, 0 in any env file), so NCCL applies its default, which merges.

What our own gate recorded at 4 HCAs:

```
transport: via NET/IB/4 via NET/IB/5
```

With four physical HCAs occupying indices 0-3, **indices 4 and 5 are two *virtual* merged
devices.** So NCCL built two merged pairs and used both -- automatic merging was on.

**Which pairs it built, we do not know.** The gate captured `ndevs=` but discarded the
`name=` field. That is the whole question:

| grouping | meaning on a ring |
|---|---|
| `rocep1s0f0+roceP2p1s0f0` | same physical port, two PCIe domains -- what working configs use |
| `rocep1s0f0+rocep1s0f1` | **two different ports -> two different neighbours** |

Our 2-HCA-era log showed the second form (`name=rocep1s0f0+rocep1s0f1`). On two nodes both
ports face the same peer, so it is harmless; **on a ring they face different neighbours**,
and NCCL then believes it has a pipe to each that it does not have.

**Fixed:** the gate now records the merged device *names* as a `vdev:*` check, so the next
run answers this directly instead of leaving it inferred.

### 3. We did not run the same benchmark

| | ours | theirs |
|---|---|---|
| harness | custom PyTorch `all_gather_into_tensor` | official MPI `nccl-tests/all_gather_perf` |
| size | 64 MiB | 32 MiB **and** 16 GiB |
| iterations | 10 | 20 |
| build | image default | NCCL 2.30.7-1, `sm_121` |

**Message size is not the explanation** -- their 32 MiB run still reached 18.64, *below*
our 64 MiB point. But "custom harness vs official binary" is an uncontrolled variable and
has to be eliminated before any conclusion is drawn.

### 4. We force all four HCAs

NVIDIA's launcher lets NCCL **discover** RoCE devices. We pass an explicit
`NCCL_IB_HCA` list, which may force an unfavourable mapping. Test both ways.

## What is ruled out

Firewall - upper-mesh addressing - MTU - memlock (`unlimited` everywhere) - CX-7 firmware
(all HCAs at NVIDIA's stated minimum `28.45.4028`) - physical link negotiation.

**But note what "200 Gb/s" in our records actually is:** `ethtool` link state, which is
negotiation, not throughput. Our measured raw single-controller RDMA is ~109 Gb/s, while
four-HCA NCCL reaches 5.80 GB/s = **46.4 Gb/s**. That gap sits *above* the cable --
discovery, merging, bootstrap, or build.

## The test

Engine **stopped**. One variable at a time; capture `NCCL_DEBUG=INFO` every run.

| # | Variable | Setting |
|---|---|---|
| 0 | baseline | our current config, for a same-day anchor |
| 1 | **bootstrap** | `wlP9s9` (Wi-Fi) addresses on all three, one common interface, MPI launcher. **No cabling needed** |
| 2 | **harness** | official MPI-enabled `nccl-tests` `all_gather_perf`, NCCL 2.30.7-1 `sm_121`, 20 iters |
| 3 | **size** | 32 MiB **and** 16 GiB, matching the forum points exactly |
| 4 | **merge** | `NCCL_IB_MERGE_NICS=0` vs default |
| 5 | **discovery** | automatic vs our explicit 4-HCA list |

Then re-add the third rank with `NCCL_IB_SUBNET_AWARE_ROUTING=1` and `NCCL_NET_PLUGIN=none`
to isolate the ring case from the pair case.

**On every run, read the `Made virtual device ... name=` line** and record which HCAs were
grouped. The gate now does this automatically (`vdev:*`).

### Reading the outcome

| Observation | Conclusion |
|---|---|
| Bootstrap change alone recovers ~18 GB/s | It was the control plane, as it was for the forum reporter. Fix the serving config the same way |
| `MERGE_NICS=0` recovers it | Automatic merging was grouping across ports. Set it explicitly and document why |
| `name=` shows cross-port pairs | Confirms the ring hypothesis directly, whatever the throughput does |
| Official harness alone changes it | Our harness was the problem -- retire it for bandwidth work |
| Nothing recovers it | The deficit is real and below all of this. *Then* it is a hardware/firmware question |

## Why this ordering

Bootstrap first because it is the one change with a **documented instance of producing
exactly this recovery, from exactly our starting number**. Merge second because we have
direct evidence NCCL is merging and no evidence about *what* it merged. Harness third
because it is uncontrolled but has no mechanism attached to it.

Nothing here should be treated as settled until the matched run exists. Our own principle
applies: **one variable, same day, same harness.**

**Related:** [#11](../../issues/11) - [`BANDWIDTH-COMPARISON.md`](BANDWIDTH-COMPARISON.md) -
[`../scripts/bwsweep.py`](../scripts/bwsweep.py) - [`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh)
