# KV pool: reconciling the init log against `/metrics` — RESOLVED

**Status:** resolved 2026-08-29 by reading the engine source in the baked image
(`dsv4-3spark:0.1.1`). No cluster time used; the live engine was scraped read-only while
the matched 2v3 sweep ran.

Supersedes the open question recorded in
[`HANDOFF-2026-08-28.md`](HANDOFF-2026-08-28.md), which said *"reconciling the two is open
work"* and provisionally treated the init log as authoritative.

---

## The apparent conflict

On the same running engine, two instruments disagreed by ~65%:

| Instrument | Tokens | Max concurrency |
|---|---:|---:|
| Engine init log (`kv_cache_utils.py:2146`) | 4,688,072 | 4.47 |
| `/metrics` `cache_config_info` | 2,839,271 | 2.7077 |

`/metrics` also reports `block_size="4"` against a launch flag of `--block-size 256`,
which made the endpoint look like it was exposing stale or internal values.

## What is actually happening

**Neither instrument is wrong, and neither is reporting a raw pool size.** From
`vllm/v1/core/kv_cache_utils.py:1788`:

```python
def get_kv_cache_capacity(vllm_config, kv_cache_config) -> tuple[int, float]:
    max_model_len = vllm_config.model_config.max_model_len
    max_concurrency = get_max_concurrency_for_kv_cache_config(vllm_config, kv_cache_config)
    return int(max_concurrency * max_model_len), max_concurrency
```

`kv_cache_size_tokens` is a **derived** figure: `int(max_concurrency × max_model_len)`.

Verified arithmetically against the live labels:

```
int(2.7077407817423147 × 1048576) = 2,839,271   ← exactly the /metrics label
4,688,072 ÷ 1,048,576             = 4.4709      ← exactly the init log's own 4.47
```

Both instruments are internally self-consistent. They differ because they compute
`max_concurrency` under different accounting.

### Why the metrics figure is lower

DeepSeek-V4 is a **hybrid** attention model — this engine reports `sliding_window="128"`
alongside full-attention layers. `_max_memory_usage_bytes_from_groups()` carries an
explicit DeepSeek-V4 branch that pads every KV cache group to a common layer-tuple layout:

> *"Even groups with fewer actual tuples still reserve the global number of tuple slots in
> the shared tensor layout."*

The `/metrics` path is **group-aware**: it charges each group for the padded layout it
actually reserves, so its concurrency — and therefore its token figure — is the
*conservative* number, constrained by the most-padded group. The init-log figure does not
carry that padding penalty.

### The `block_size="4"` label

Not stale, and not the launch flag. `vllm/v1/engine/core.py:309` deliberately overwrites
it after KV groups are built:

```python
vllm_config.cache_config.block_size = min(
    g.kv_cache_spec.block_size for g in kv_cache_groups
)
```

On a hybrid model that is the **minimum across groups**, not the requested value. The
companion labels say so directly: `_block_size_resolved="True"` and
`user_specified_block_size="True"` — the engine is recording that the user asked for 256
and that a resolved value replaced it. `--block-size 256` is not being ignored.

## Which number to quote

| Question | Use | Why |
|---|---|---|
| "How much KV did this boot allocate?" | **init log**, 4,688,072 | Closest to the physical pool |
| "How many 1M-token requests can it really hold?" | **`/metrics`**, 2.71x | Group-aware; accounts for hybrid padding |
| Any public claim | **both, labelled** | They measure different things |

**Always state the instrument AND the MTP depth.** MTP depth changes CUDA-graph capture
size, which changes the memory left for KV — the `~2.49M` figure in
[`20260827-issue25-profile-b`](../results/20260827-issue25-profile-b/) is correct *for that
bundle*, which ran `MTP_NUM_TOKENS=5`.

## Correction to the published record

The 2v3 benchmark matrix cites a KV pool of **4,457,627 tokens** for TP=3 and
**1,711,307** for TP=2, a "2.6x" advantage. Those are init-log-family numbers. A
group-aware reading of the same engines would give lower absolute figures for both arms.
The **ratio** is the defensible part of that row; the absolute values must carry their
instrument. This does not change any decode or TTFT measurement.

## What remains open

Whether `get_max_concurrency_for_kv_cache_config` and the init-log path diverge for any
reason *beyond* hybrid group padding has not been traced line-by-line. The arithmetic
above closes exactly, and the padding branch is explicitly DeepSeek-V4-specific, so
padding is the established cause — but a full trace of both call paths would make it
airtight.
