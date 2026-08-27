# Prefill: 2 nodes vs 3 nodes on healthy fabric — 2026-08-25

The question: now that the fabric is fixed, does the third node make prefill faster?

**Answer: no. They are at parity.**

| input tokens | TP=2 | TP=3 | 3-node gain |
|---:|---:|---:|---:|
| 1,024 | 1,913.1 | 2,022.6 | +5.7% |
| 8,192 | 2,080.6 | 2,069.5 | **−0.5%** |
| 32,768 | 2,065.8 | 2,094.9 | +1.4% |

Server-side tok/s, median of 3. Within ±2% at the depths that matter; the +5.7% at 1,024
is the smallest, most fixed-overhead-dominated shape and should not be read as a trend.

## Why this is a fair comparison

- **Same config on both arms.** Production profile — `MAX_MODEL_LEN=1048576`,
  `MAX_NUM_SEQS=16`, `MTP_NUM_TOKENS=5`, `GPU_MEMORY_UTILIZATION=0.80`. Node count is the
  only variable.
- **Same harness, unmodified.** Upstream anemll's `benchmark_prefill.py`, byte-identical
  (`md5 f5c3269d…`) to the run that produced the TP=3 numbers, same seed (4104).
- **Real prefill, not cache.** Every trial reports `computed=` equal to the full input
  size, with no `cache-hit` field. Client and server agree within 1%.
- **Fabric gated immediately before**, engine stopped so NCCL bandwidth could actually be
  measured: 12/12 checks pass, pairs at 4.60–4.67 GB/s (`fabric-gate.json`).

## What this settles, and what it doesn't

**Settles:** the third node does not buy prefill throughput. Combined with the
deep-concurrency re-run (where 2-node reached first token **1.35x sooner**), the case for
the third node does not rest on prefill.

**Does not settle:** decode. The healthy-fabric decode numbers in `HANDOFF.md` (85.6 tok/s
cc=1, 491.0 cc=16) are **TP=3 only** — there is no matched TP=2 decode arm on healthy
fabric. The "+8–17% per-stream at long context" claim that justifies three nodes is still
a **degraded-fabric** measurement and remains open under issue #14.

> **Closed 2026-08-26** (forward pointer only; nothing above is edited). The matched decode
> arms landed: parity below 32K, **+33.6% at 131K**, +17.9% at 262K — so "+8–17% from 2K
> upward" is retracted, and the case for the third node rests on **decode past ~100K**, not
> on prefill and not on short contexts. That result also finds TTFT at depth favouring
> **two** nodes, which is the same direction as the prefill parity measured here.
> [`../20260826-decode-depth-2v3/`](../20260826-decode-depth-2v3)

## The prior fabric state changes the sign

Same harness, same depths, on the degraded fabric (2026-08-24):

| input tokens | TP=2 degraded | TP=3 degraded | TP=3 healthy |
|---:|---:|---:|---:|
| 1,024 | 1,111.8 | 1,063.0 | 2,022.6 |
| 8,192 | 1,110.2 | 1,074.9 | 2,069.5 |
| 32,768 | 1,105.9 | 1,034.3 | 2,094.9 |

Two things worth noting. On the degraded fabric **3 nodes were 4–6% *slower* than 2** at
every depth — the handicap fell on the 3-node arm, exactly as issue #14 predicted. And the
fix took TP=3 prefill from 1,034 → 2,095 at 32K (**2.02x**), turning a curve that
*degraded* with depth into one that is flat-to-rising. That shape change is the signature
of a communication bottleneck being removed.

## Mechanism — this result is expected, per vLLM maintainers

vLLM maintainer `andoorve` on when pipeline parallelism beats tensor parallelism
([discussion #10118](https://github.com/vllm-project/vllm/discussions/10118)): PP is
preferable with *"poor interconnect, **more communication volume from prefills**,
cross-node."*

TP allreduce volume during prefill scales with **batched tokens × hidden size** — i.e.
linearly in chunk size — while decode's scales with batch size only. Prefill is therefore
where cross-node TP breaks down first, and adding a third rank to a TP group adds
collective cost that the extra compute has to earn back. At these depths it does not.

vLLM's own guidance is that TP should stay **within** a node and PP should span nodes
([Parallelism and Scaling](https://docs.vllm.ai/en/stable/serving/parallelism_scaling/)).
With 1 GPU per Spark that means TP=1/PP=3, which we measured as blocked (MTP + a DSA
stride constraint) — see [`../../docs/PP3-PIPELINE-PARALLEL.md`](../../docs/PP3-PIPELINE-PARALLEL.md).

## ⚠ Read before drawing conclusions about node count

Our NCCL bandwidth may itself still be far below what this hardware does. Independent
measurements on the **identical 3-Spark ring** report **18–21 GB/s** allgather
([NVIDIA forum](https://forums.developer.nvidia.com/t/test-the-sample-about-connect-three-dgx-spark-in-a-ring-topology/365160));
we measure **3.25 GB/s** 3-rank and ~4.6 GB/s pairwise. In that thread a user first
measured **2.86 GB/s** — very close to ours — and traced it to NCCL binding the wrong
interface addresses.

If that gap is real and closable, prefill is currently communication-starved on **both**
arms and this parity result may not survive. Prime suspect is `NCCL_IB_MERGE_NICS` (GB10
exposes each physical port as two logical 100G interfaces), though every published
merge result is from **2-node** setups where both ports face the same peer — in a ring
they face different neighbours, and no source we found addresses that case.

**Do not tune vLLM prefill flags before resolving the fabric number.** Note also that
`HANDOFF.md` currently records `NCCL_IB_MERGE_NICS` as a no-op — a conclusion reached on
the degraded fabric, so it belongs on the issue #14 suspect list too.

## Files

| file | what |
|---|---|
| `tp2_prefill.json` / `.txt` | the TP=2 arm, full per-trial output |
| `fabric-gate.json` | `scripts/fabric_gate.sh --nccl=pairs` taken immediately before |

The TP=3 arm lives at [`../20260825-fabric-fix/anemll_fresh.txt`](../20260825-fabric-fix).

---

## Extended 2026-08-26: what happens past 32K

This page only reaches 32,768 tokens. Two measurements taken on 2026-08-26 extend the
depth axis, and the answer changes sign.

**TTFT is prefill latency**, so the depth sweep's TTFT column measures the same physical
work by a second instrument ([`../20260826-decode-depth-2v3/`](../20260826-decode-depth-2v3),
median of 7, matched arms, fabric gated 33/0):

| depth | TP=2 TTFT | TP=3 TTFT | 3-node | implied TP=3 tok/s |
|---:|---:|---:|---:|---:|
| 2,036 | 1.1 s | **0.9 s** | **−18%** (faster) | 2,213 |
| 8,081 | 3.9 s | **3.3 s** | **−15%** (faster) | 2,419 |
| 32,268 | 16.0 s | **14.6 s** | **−9%** (faster) | 2,206 |
| 129,006 | **72.4 s** | 77.1 s | +6.5% (slower) | 1,673 |
| 257,993 | **158.4 s** | 181.6 s | **+14.6%** (slower) | 1,421 |

**The third node helps prefill below ~32K and costs 6–15% past ~100K.** That is
consistent with this page's parity finding — the shallow depths here sit in the
"slightly faster" region, within the ±2% this page measured with its own harness — and
it adds the crossover this page could not see.

**Prefill rate declines with depth on both configurations**, from ~2,400 tok/s at 8K to
~1,400 at 258K, and to ~700 near the 1M ceiling
([`../20260826-near-ceiling-prefill/`](../20260826-near-ceiling-prefill), preliminary).
That decline is a property of attention cost at depth, **not** of node count.

> **Caveat on instrument.** The TTFT figures come from a decode-focused harness with a
> prose filler prompt, not from `benchmark_prefill.py`'s deterministic token-ID pool.
> Treat them as corroboration of the *direction*, not as a replacement for a designed
> prefill run. **A matched `benchmark_prefill.py` arm above 32K does not exist yet** —
> that is the remaining gap on this axis.
