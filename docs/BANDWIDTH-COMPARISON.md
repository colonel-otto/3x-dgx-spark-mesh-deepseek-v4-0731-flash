# The bandwidth gap: RESOLVED — there was never one

> [!IMPORTANT]
> **Settled 2026-08-26 by controlled measurement ([#18](../../issues/18)).**
> **There is no fabric deficit.** The same unchanged 3-rank configuration that read
> **5.80 GB/s** under our custom harness reads **23.92 GB/s** under the official
> `nccl-tests all_gather_perf`. We match or exceed the published figures we were chasing.
>
> This page previously argued the gap was a metric-convention artifact (wrong), then that
> it was a real ~3.2x hardware deficit with HCA pairing as the lead (also wrong). Both
> corrections are kept below, because the sequence is the lesson.

---

## The result

`all_gather_perf`, NCCL 2.30.7, `-n 20`, engine stopped, busbw out-of-place — the same
collective, convention and iteration count the published figures use.

| variant | world | bootstrap | 32 MiB | **16 GiB** |
|---|---:|---|---:|---:|
| A baseline (our live config) | 3 | fabric | 16.85 / 15.74 † | **23.92** |
| B bootstrap over management | 3 | `wlP9s9` | 16.14 | **23.94** |
| C `NCCL_IB_MERGE_NICS=0` | 3 | `wlP9s9` | 17.91 | **23.86** |
| D automatic HCA discovery | 3 | `wlP9s9` | 16.72 | **23.85** |
| E world = 2 | 2 | `wlP9s9` | 16.22 | 22.30 |

† First A run read 4.81 — a cold-start outlier (4652 µs vs ~1300 µs on repeat). Re-run
twice: 16.85 and 15.74. Recorded rather than discarded.

**Against the published reference:** 18.70 @32 MiB and 20.84 @16 GiB. We are **above both**
at 16 GiB, and in the same band at 32 MiB.

**The ~24 GB/s PCIe ceiling is confirmed.** Every 3-rank variant pins to it — consistent
with both 200G ports sharing two PCIe Gen5 x4 lanes (~252 Gb/s).

## Which variable moved the number: none of them

At 16 GiB the four 3-rank variants span **23.85–23.94 GB/s — a 0.4% spread.**

| change | effect |
|---|---:|
| bootstrap: fabric → management | **+0.1%** |
| NIC merging: on → off | **−0.3%** |
| explicit 4-HCA list → auto-discovery | **−0.4%** |

**The only variable that mattered was the harness.** Same config, same fabric, same day:
5.80 GB/s custom vs 23.92 official.

## Every hypothesis we held, measured

### ❌ "Bootstrap topology is the strongest lead"

Refuted, and the change was verified real rather than assumed: NCCL bound to
`192.168.100/101/102.x` in variant A and `192.168.1.x` in B, with data staying on
`NET/IB` in both. **Effect: +0.1%.**

The reasoning was sound — the one public 3-Spark result did recover from 2.86 GB/s by
changing its bootstrap. But *their* fix does not imply *our* fault. We inferred a shared
cause from a shared symptom.

### ❌ "Cross-port NIC merging breaks the ring"

The mechanism was real and confirmed present. NCCL does merge across two different
physical ports, verbatim on all three nodes:

```
Made virtual device [4] name=rocep1s0f0+rocep1s0f1     speed=400000 ndevs=2
Made virtual device [5] name=roceP2p1s0f0+roceP2p1s0f1 speed=400000 ndevs=2
```

On this ring `f0` and `f1` genuinely face different neighbours, so this is exactly the
pairing we predicted would be harmful. **Variant C disabled merging entirely: 23.86 vs
23.94 — harmless.**

A correct mechanism is not a correct diagnosis.

### ❌ "There is a rank-count penalty"

3-rank (**23.92**) slightly *exceeds* 2-rank (**22.30**). The "deficit grows with rank
count" pattern — which I offered as evidence for the ring hypothesis — was itself a
harness artifact.

### ✅ "Our formula matches theirs"

Correct, and it is why the harness difference stayed invisible so long. Both compute
`busbw = algbw * (n-1)/n`. The formula was never the problem; **what was being measured**
was.

## Why our harness read low

`results/20260824-seqs32-nccl/agbench.py` was built to mirror the **vLLM MTP allgather
shape** (`seqs16` = 4.14 MB) — the right instrument for "does our workload run well". It
was never a peak-bandwidth benchmark, and it tops out at a 67 MB input.

We then compared its top point against someone else's peak-bandwidth figure. Differences
that matter: 10 iterations vs 20, no warmup discipline, in-place vs out-of-place
convention, PyTorch collective overhead, and no `#wrong` correctness column.

**The instrument was fine. Using it outside its purpose was not.**

## What we got wrong, in order

1. **Claimed the gap was a convention artifact.** Refuted by nccl-tests source — same
   collective, same formula.
2. **Estimated a 48.5 GB/s ceiling from line rate.** PCIe caps it near 24. *(This
   correction was right, and today's data confirms it.)*
3. **Concluded a real ~3.2x hardware deficit.** There is none.
4. **Named cross-port merging as the mechanism.** Present, confirmed, and harmless.
5. **Named bootstrap as the strongest lead.** Worth +0.1%.
6. **Never ran the official binary.** Four wrong conclusions, days of investigation, and
   the answer took one matched run.

**The through-line:** every hypothesis was plausible, mechanistically coherent, and
consistent with the numbers we had. None survived contact with the control. We kept
reasoning about *why* the number was low instead of first checking *whether* it was.

## The lesson worth keeping

> When your measurement disagrees with a published one, **suspect the measurement before
> the machine** — and reach for the reference implementation early. A custom harness is
> evidence about your workload; only the standard tool is evidence about your hardware.

Corollary, which cost most of the delay: **a mechanism that is real, present, and predicts
the observed pattern can still be the wrong explanation.** Cross-port merging was all
three, and irrelevant.

## Two traps found on the way

**The version trap is live and would have silently invalidated everything.** The binary
compiles against 2.30.7 headers, but the loader resolves `libnccl.so.2` to the *system*
2.28.9 copy. The first run printed `nccl-headers=23007 nccl-library=22809`. Fixed with
`LD_LIBRARY_PATH`, then verified `23007/23007` independently **on all three nodes** — a
per-rank mismatch would have been undetectable in the result. See
[`NCCL-TESTS-BUILD.md`](NCCL-TESTS-BUILD.md).

**MPI's own OOB cannot use the fabric.** It is a set of point-to-point `/30` links, not a
full mesh — spark2 has no route to `192.168.110.1`. MPI is launcher-only on management;
`NCCL_SOCKET_IFNAME` is the actual variable under test.

## Evidence

[`../results/20260826-nccl-controlled/`](../results/20260826-nccl-controlled) — 12 runs,
24 logs, `summary.json`, `run_variant.sh`, `Dockerfile.ntests`.

Every run: NCCL **2.30.7**, `#wrong=0`, transport `via NET/IB/*`, **zero `NET/Socket`
occurrences**. No run void.

## Sources

- [nccl-tests PERFORMANCE.md](https://github.com/NVIDIA/nccl-tests/blob/master/doc/PERFORMANCE.md) — the formulas
- [Forum 365160](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160) — the 3-rank ring reference
- [Forum 350417](https://forums.developer.nvidia.com/t/connectx-7-nic-in-dgx-spark/350417/61) — PCIe lanes, no GPUDirect

**Related:** [#18](../../issues/18) · [`NCCL-TESTS-BUILD.md`](NCCL-TESTS-BUILD.md) ·
[`../results/20260825-upper-mesh/`](../results/20260825-upper-mesh)
