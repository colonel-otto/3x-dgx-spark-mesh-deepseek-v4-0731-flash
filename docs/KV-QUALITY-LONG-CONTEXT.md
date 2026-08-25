# NVFP4 KV-cache output quality under long context

> [!NOTE]
> **SINGLE-ARM RESULT.** Quality was measured clean to 464K on `nvfp4_ds_mla` — but with
> **no comparison against `fp8_ds_mla`**, which is what both official recipes specify.
> `nvfp4_ds_mla` has no published accuracy evaluation anywhere, and the two dtypes are
> **memory-identical** on DeepSeek-V4 (both use the 584-byte sparse-MLA envelope), so the
> choice is free and should be made on evidence. Tracked as [#16](../../issues/16).

Experiment date: 2026-08-24 (UTC). Cluster: 3-node TP=3, `MAX_MODEL_LEN=1048576`,
`MTP_NUM_TOKENS=5`, `MAX_NUM_SEQS=16`, `GPU_MEMORY_UTILIZATION=0.85`,
`--kv-cache-dtype nvfp4_ds_mla`, KV pool 5,444,869 tokens. Image
`ghcr.io/anemll/dspark-vllm-gx10:0.1.1`, vLLM `0.25.2.dev0+g752a3a504.d20260714`.

Answers issue #12. **This is a correctness test, not a benchmark** — throughput is
deliberately not the subject.

## Why this was run

External advice received 2026-08-24 recommended dropping KV quantization "to keep maximum
performance." The *performance* half of that claim is measured false and was dismissed in
[`MTP5-1M-AND-UPSTREAM-COMPARISON.md`](MTP5-1M-AND-UPSTREAM-COMPARISON.md): upstream A/B'd
`fp8_ds_mla` against `nvfp4_ds_mla` and got 41.4 versus 41.5 tok/s peak, with no acceptance
difference — *"a context lever, not a speed lever."*

But the same upstream repository carries a separate, credible **quality** warning: 4-bit KV
*"can collapse into salad under long, heavy agentic context"* while fp8 KV stays clean, and
it recommends fp8 *"when clean output under concurrency matters more than max context."*
That warning was untested here, and it mattered more once we raised `MAX_MODEL_LEN` to 1M.

## Method

Two independent signals per depth, so a failure is not a matter of taste.

**1. Needle retrieval.** Three unique access codes are buried at 10%, 50%, and 90% of the
context. The model either reproduces the exact token or it does not. Three positions rather
than one, because position-dependent degradation is the shape KV-cache damage would take —
a single mid-context needle could pass while the early context had already rotted.

**2. Garble detection**, targeting the specific failure mode described upstream: CJK
characters, Unicode replacement chars, stray BOS/special tokens, and pathological repetition
in the reply.

Filler is varied technical prose that never mentions a needle value, so a correct answer
cannot be guessed from context. It is deliberately **not** repeated text — repeated filler
is highly compressible for the sparse-attention indexer and would make the test easier than
real use. Temperature 0.

Harness: [`../results/20260824-kv-quality/kvquality.py`](../results/20260824-kv-quality/kvquality.py).

## Results

| Requested depth | Actual prompt tokens | Latency | Needles | Garble | Result |
|---:|---:|---:|---|---|---|
| 2,000 | 1,947 | 2.6 s | 3/3 | none | **PASS** |
| 8,000 | 7,519 | 16.0 s | 3/3 | none | **PASS** |
| 32,000 | 29,774 | 33.1 s | 3/3 | none | **PASS** |
| 64,000 | 59,444 | 69.2 s | 3/3 | none | **PASS** |
| 128,000 | 118,798 | 146.4 s | 3/3 | none | **PASS** |
| 200,000 | 185,579 | 230.1 s | 3/3 | none | **PASS** |
| 350,000 | 324,682 | 443.0 s | 3/3 | none | **PASS** |
| 500,000 | 463,792 | 603.3 s | 3/3 | none | **PASS** |

**Clean through 463,792 real prompt tokens.** Every needle at every position, no garble of
any kind, no degradation trend as depth grows.

500K is the depth upstream explicitly cites (*"Aiden serves 500K context on fp8"*) as the
case for preferring fp8. On this stack, NVFP4 KV is clean there.

## Conclusion

**Keep `nvfp4_ds_mla`.** As of this measurement the case for switching is zero on both axes:

- **Performance:** measured identical to fp8 upstream. Not a speed lever.
- **Quality:** no degradation observed to 463K tokens, including at 500K where the upstream
  warning is aimed.

NVFP4 is what buys the 5,444,869-token KV pool and the 1M context. Do not trade that away
without a reproducing failure.

## What this does NOT establish

**Concurrency is untested.** Every measurement above is a single request against an idle
engine. The upstream warning specifically pairs long context *with* concurrency ("clean
output **under concurrency**"), and KV pressure is a plausible trigger this test never
applies. A concurrent long-context run is the obvious next probe.

**Agentic structure is untested.** The filler is prose, not tool calls, JSON state, or
multi-turn history. Real agentic context has different token statistics, and the upstream
report describes *"heavy agentic context"* specifically.

**One prompt shape, one seed, temperature 0.** No repetitions per depth; a rare
intermittent failure would not appear. Temperature 0 is the most favourable sampling case.

**Not an A/B against fp8.** No fp8 control was run, because nothing failed — there was no
effect to attribute. If a failure is ever found, the A/B at matched context is the required
next step before blaming the KV dtype.

**Depths above ~464K were not measured.** A 700K/900K run was started and deliberately
abandoned: 500K is the depth the upstream warning targets, it passed cleanly, and that was
accepted as sufficient to settle the question. If a future workload genuinely runs past
~500K, rerun `kvquality.py 700000 900000` — do not assume the clean trend continues to the
1,048,576 ceiling.

## Reproducing

```bash
# Confirm the endpoint is idle first; one in-flight request skews the timing rows.
curl -s http://localhost:8100/metrics | grep -E '^vllm:num_requests_running\{'

python3 kvquality.py 2000 8000 32000 64000 128000 200000 350000 500000
```

Expect roughly 10 minutes per request at 500K. A FAIL prints the missing needles and the
first 300 characters of the reply, so the failure mode is visible rather than merely
counted.
