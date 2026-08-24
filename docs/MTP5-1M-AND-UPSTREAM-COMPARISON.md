# MTP=5 isolation, 1M context, and comparison against two upstream repositories

Experiment date: 2026-08-24 (UTC). Cluster: sparkmain (rank 0) + spark1 (rank 1) +
node2 (rank 2, reachable as both `spark2` and `spark-sep`). Image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM `0.25.2.dev0+g752a3a504.d20260714`,
model `/models/dsv4-abliterated` (DeepSeek-V4-Flash-0731 abliterated), TP=3 with the
`o_groups` padding patch applied at launch.

This branch answers three questions:

1. What does raising `MAX_MODEL_LEN` to the model's full 1,048,576 cost?
2. Is `MTP_NUM_TOKENS=5` better than `4`? — the control
   [`TP3-TUNING.md`](TP3-TUNING.md) explicitly said was missing.
3. How do we compare against the two public DGX Spark repositories?

## 1. Headline results

| Change | Verdict |
|---|---|
| `MAX_MODEL_LEN` 460,800 → **1,048,576** | **Free.** KV pool *grew*; aggregate throughput unchanged within spread. |
| `MTP_NUM_TOKENS` 4 → **5** | **Better for single-stream decode** (+8–13% on structured/code). Aggregate unchanged. |

## 2. The 1M context change

`MAX_MODEL_LEN=1048576` at `MAX_NUM_SEQS=16`, `GPU_MEMORY_UTILIZATION=0.85`, MTP=4:

| | 460,800 | 1,048,576 |
|---|---:|---:|
| GPU KV cache size | 3,565,267 tokens | **5,428,503 tokens** |
| Available KV memory | 37.36 GiB | 37.8 GiB |
| Max concurrency @ full context | 7.74x | 5.18x |
| Graph capture | 11 s / 0.82 GiB | 19 s / 1.76 GiB |
| Aggregate @ cc=16 | 370.9 tok/s | 353.1 tok/s |

The KV pool grew by 52% because the block allocator sizes to the model's real
capacity, not because more memory was granted. The −5% at cc=16 is inside this
cluster's run-to-run spread (see §5) and is not a reliable regression signal.

**There is no reason to run this cluster at 460,800.** 1M context is the model's
calibrated YaRN ceiling and it costs nothing measurable here.

## 3. MTP=5 versus MTP=4 — the isolated control

[`TP3-TUNING.md`](TP3-TUNING.md) recorded MTP=5 at `seqs=16` against MTP=4 at
`seqs=8` and correctly refused to call four tokens optimal, because two variables
moved at once. This is the matched run: **only `MTP_NUM_TOKENS` changes.** Both
sides at `MAX_MODEL_LEN=1048576`, `MAX_NUM_SEQS=16`, `GPU_MEMORY_UTILIZATION=0.85`,
temperature 0, three repetitions each, after warm-up to steady state.

### Single-stream decode and draft acceptance

Median of 3 repetitions. `accept-len` is accepted tokens per draft step, out of k.

| Prompt type | MTP=4 tok/s | MTP=5 tok/s | Δ | MTP=4 accept% | MTP=5 accept% | MTP=4 len | MTP=5 len |
|---|---:|---:|---:|---:|---:|---:|---:|
| count300 | 86.8 | **93.8** | **+8.1%** | 100.0 | 100.0 | 4.00/4 | 5.00/5 |
| json60 | 83.3 | **92.3** | **+10.8%** | 98.1 | 98.5 | 3.93/4 | 4.93/5 |
| code (BST) | 76.3 | **86.1** | **+12.8%** | 86.1 | 85.5 | 3.44/4 | 4.28/5 |
| prose | **40.8** | 34.4 | −15.7% | 32.0 | 21.1 | 1.28/4 | 1.05/5 |

**The fifth draft token lands.** On deterministic and structured content, acceptance
*percentage* is unchanged while accepted *length* rises almost exactly one full token
(4.00→5.00, 3.93→4.93, 3.44→4.28). That is the marginal draft being accepted, not
merely offered — the earlier concern that "the marginal fifth draft appears to have low
acceptance" is not supported once sequence count is held constant.

Prose is the exception and moves the other way. Prose acceptance is low at any k
(21–32%), so the extra draft is usually rejected and its cost is not recovered.
Prose is also the noisiest workload here (33.1–39.6 tok/s across MTP=5 reps versus
under 1% spread on code/json/count), so the −15.7% is the least trustworthy row in the
table. It is directionally real but its magnitude is not settled.

### Aggregate throughput — unchanged

| Concurrency | MTP=4 | MTP=5 |
|---:|---:|---:|
| 1 | 77.0 | 71.4 |
| 2 | 119.3 | 116.0 |
| 4 | 171.9 | 176.6 |
| 8 | 269.8 | 261.0 |
| 12 | 297.0 | 309.4 |
| 16 | **374.2** | 349.5 |
| 24 | 326.8 | 285.8 |

Medians of 3. Every difference here is within the spread of the runs that produced
it — several levels flagged 12–34% spread. **Aggregate throughput does not
distinguish these two settings**; the single-stream table above does, because at
temperature 0 with a fixed prompt the acceptance counters are near-deterministic.

### Recommendation

`MTP_NUM_TOKENS=5` for interactive/agentic use, where single-stream latency is what a
caller feels and the workload is structured. The gain is 8–13% on exactly the content
an agent emits. Keep `4` only if the dominant workload is long-form prose.

Cost: graph capture rises from 80 to 96 (`max_num_seqs × (k+1)`), measured at
1.20 GiB / 8 s versus 1.76 GiB / 19 s — no OOM at `GPU_MEMORY_UTILIZATION=0.85`.
KV pool is unaffected (5,433,516 versus 5,412,285 tokens).

**MTP=5 boots and runs cleanly on this stack.** The 2-node repository's warning that
`k` must be `<= 5` or a multiple of 5 is satisfied at exactly 5; we did not test 7 or 10.

## 4. Comparison against the two public repositories

Both comparisons below were produced by running **the upstream repository's own
benchmark script, unmodified, against our endpoint** — not by quoting our harness
against their published table.

### 4a. `localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark` (3 nodes, TP=3)

The closest comparison available: same hardware count, same parallelism, same
`o_groups` padding patch. Their `scripts/benchmark_tp3.py`, 256 tokens, temperature 0.
Their published column is `max_num_seqs=16`, matching ours.

| Concurrency | Ours (460,800, MTP=4) | Theirs (1M, MTP=5) | Δ |
|---:|---:|---:|---:|
| 1 | 77.6 | 75.3 | +3% |
| 2 | 108.7 | 113.4 | −4% |
| 4 | 181.3 | 177.2 | +2% |
| 8 | 245.0 | 291.7 | −16% |
| 12 | 303.2 | 369.8 | −18% |
| 16 | 370.9 | **431.3** | −14% |
| 24 | 311.4 | 367.7 | −15% |

We track them at low concurrency and sit ~15% behind from cc=8 up. Saturation lands at
cc=16 for both, then falls — same curve shape.

Their headline **618 tok/s requires `max_num_seqs=32`**, which we have not tested.
Quoting 618 against any number in this document would be comparing different
configurations. At the setting we both run, their figure is 431.

Untested config deltas that remain between us: their `max_num_seqs=32` profile, and
their model is a different abliterated checkpoint
(`keys-DeepSeek-V4-Flash-GA-0731-Dspark-Abliterated-32-32`).

### 4b. `tonyd2wild/...-2x-DGX-Spark` (2 nodes, TP=2)

Their `benchmarks/bench_full.py`, unmodified, against our endpoint (460,800, MTP=4):

| Metric | Ours (TP=3) | Theirs (TP=2) | Δ |
|---|---:|---:|---:|
| Decode peak | **91.1** | 84.3 | +8% |
| Decode mean (5 types) | **76.0** | 67.6 | +12% |
| Aggregate @ c6 | **200.1** | 197.3 | +1% |
| Prefill @ 100K | 1,512 | **2,639** | **−43%** |

Decode is ahead; prefill is the one real deficit and is **unexplained**. It is not
node count and not the shared-expert bug (§4c).

**Prefill is highly sensitive to prompt content.** Measured on the same server, same
day: a random-word filler gives ~1,000 tok/s flat at 25K/50K/100K, while their
repeated-sentence filler gives 1,512 at 100K — a 51% swing from prompt content alone.
Their 2,639 is therefore a best case, not a general prefill rate. Note also that their
harness targets 100,000 *characters*, which tokenizes to ~78K tokens, so "100K" is not
100K tokens on either side.

### 4c. Patch 4 (shared expert) — not applicable to this image

The 2-node repository documents a draft-loader bug that silently drops the DSpark
draft's always-on shared expert, costing ~70% of decode speed (32.7 → 55.4 tok/s,
acceptance 25.7% → 60.2%) with no visible output-quality symptom. **Our image is not
affected**, verified three ways:

1. Their vLLM 0.21.1rc1 mapping was anchored on the full `.shared_experts.wN`
   segment. Our 0.25.2 loader
   (`vllm/models/deepseek_v4/nvidia/dspark.py:370-371`) uses a generic substring
   rule, `("gate_up_proj", "w1", 0)`, which matches `"w1" in name` and therefore
   catches `.shared_experts.w1`/`.w3` and merges them into `gate_up_proj`. Simulated
   against the live table: both map correctly, and `markov_w1` is excluded by the
   `is_layer_param` guard rather than colliding.
2. Startup logs show `DSpark draft model loaded: 96 params` with **zero** "Skipping
   unknown DSpark weight" lines.
3. Live acceptance on structured content is 98.1–98.5%, far above both their broken
   (25.7%) and fixed (60.2%) states.

Applying the patch to this image would be redundant.

## 5. Methodology and honesty notes

**Warm-up is not optional, and one pass is not enough.** After every restart the first
sweep read cc=16 at 206–222 tok/s against a steady-state 340–375. A second sweep was
still climbing in one case (330 → 346). All measured numbers here come after
warm-up sweeps were repeated until two agreed. A cold number is roughly 40% low —
larger than the ~30% the 2-node repository reports.

**Run-to-run spread on this cluster is material.** Several concurrency levels flagged
12–34% spread even warm, which is why §3 rests on the near-deterministic acceptance
counters rather than on aggregate throughput, and why the −5% in §2 is not called a
regression.

**Config discovery correction.** The `dsv4` launcher and `~/bin/dsv4` symlink describe
a 2-node TP=2 deployment and read `config/head.env`; `dsv4.service` invokes them. The
running 3-node cluster was in fact launched from `config/tp3.env` on each node,
confirmed via the containers' `com.docker.compose.project.environment_file` label. All
edits in this experiment were applied to `config/tp3.env` on all three ranks. **The
service scripts are stale for this deployment** and would start the wrong topology.

**Not claimed here.** `max_num_seqs=32` is untested on our cluster; capture size is
derived from it (`32 × 6 = 192`) and our `GPU_MEMORY_UTILIZATION` is 0.85 against the
3-node repository's 0.80, so it carries real OOM risk and needs its own run.
`NCCL_IB_MERGE_NICS=1` is absent from our environment and is a live lead for the
prefill gap, but our fabric is switched rather than back-to-back QSFP, which is the
case that note was written for. No prefill figure exists for TP=3 in either public
repository, so the prefill deficit in §4b is measured only against a 2-node deployment.

## 6. Raw data

On the head node: `/tmp/ours_tp3_1m_mtp5.json`, `/tmp/ours_tp3_1m_mtp4_matched.json`,
`/tmp/ours_tp3_1m_mtp4.json`, `/tmp/ours_tp3.json`. Harnesses: `/tmp/bench_tp3.py`
(3-node repo), `/tmp/bench_full.py` (2-node repo), `/tmp/accept.py` (acceptance
instrumentation written for this experiment).
