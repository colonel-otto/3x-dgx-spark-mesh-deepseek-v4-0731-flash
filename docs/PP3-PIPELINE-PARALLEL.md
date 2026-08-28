# 3-Spark pipeline parallelism (TP=1 / PP=3 / EP off)

Captured 2026-08-21. Raw data and the exact configs that ran:
`results/20260821T133000Z-3spark-pp3/`.

**Summary: the fast MoE kernel survives pipeline parallelism. PP is blocked anyway, by
two constraints that have nothing to do with the MoE kernel. No throughput number was
produced — this is a blocked result, not a slow one.**

This is the architecture PR #1 originally proposed (`TP=1 / PP=3`, speculation off) and
which PR #3 never ran, having pivoted to expert parallelism. It matters because PR #3
established that enabling EP costs 2.5x by forcing `flashinfer_b12x` off. PP was the
obvious way to use three nodes *without* paying that.

---

## 1. The result that matters

```
(Worker_PP0) INFO [mxfp4.py:426] Using 'B12X_MXFP4' Mxfp4 MoE backend.
```

That is a live `TP=1 / PP=3 / nnodes=3` engine across all three Sparks, loading the
Spark-tuned kernel that expert parallelism cannot use.

This is source-predictable, and it sharpens PR #3's conclusion. The gate at
`b12x_mxfp4_moe.py:596-603`:

```python
@staticmethod
def _supports_parallel_config(moe_parallel_config) -> bool:
    return (
        not moe_parallel_config.use_ep
        and moe_parallel_config.ep_size <= 1
        and not moe_parallel_config.use_all2all_kernels
        and not moe_parallel_config.enable_eplb
    )
```

and `use_ep` itself, at `fused_moe/config.py:1190`:

```python
use_ep = (dp_size_ * pcp_size_ * tp_size_ > 1
          and vllm_parallel_config.enable_expert_parallel)
```

`pipeline_parallel_size` appears in **neither**. With `TP=1 / DP=1 / PP=3` and no
`--enable-expert-parallel`, `use_ep=False` and `ep_size=1`, so the gate passes.

### Why this changes the wording of PR #3

PR #3 is right that the kernel is the cost. It should not be read as "B12X is
incompatible with multi-node execution" or "MXFP4 and MoE parallelism are fundamentally
exclusive". The accurate scope is:

> The B12X MXFP4 kernel in this DGX Spark runtime does not support **expert**
> parallelism, which forces EP onto a substantially slower MoE backend.

That is a current software limitation — which means it is fixable, and worth watching
upstream rather than treating as physics.

---

## 2. Blocker one: MTP cannot coexist with PP

First launch, MTP left on:

```
NotImplementedError: Pipeline parallelism is not supported for this model.
Supported models implement the `SupportsPP` interface.
```

Raised from `create_speculative_config` → the **draft** model's
`verify_with_parallel_config` (`config/model.py:1223`).

| Class | File | Implements `SupportsPP` |
|---|---|---|
| `DeepseekV2ForCausalLM` (target) | `deepseek_v2.py:1777-1779` | **yes** |
| `DeepSeekMTP` (draft) | `deepseek_mtp.py:223` | **no** |

DSv4-Flash is served by the `deepseek_v2` architecture, which supports PP fine. The
DSpark MTP speculator does not, and the check does not exempt draft models.

> [!NOTE]
> **UPDATE 2026-08-28 — this blocker is a wrong question, not a wall.** A public patch
> set ([allover326/deepseek-v4-cmp170hx](https://github.com/allover326/deepseek-v4-cmp170hx),
> pinned `3dd2d88`) runs DSpark speculation under PP on 4x sm_80 by observing that the
> draft is built on the **last PP rank only** and runs whole there — it is a `pp_size=1`
> model regardless of the target's split, so it never needed `SupportsPP`. Their fix is a
> one-line `draft_parallel_config.pipeline_parallel_size = 1` plus a draft-token broadcast
> to non-last ranks (without which acceptance ~0 and output is silently garbage). Unported
> and unrun on GB10/B12X. Verification anchors and caveats:
> [`HANDOFF-ENGINE-CLONE-REVIEW-2026-08-28.md`](HANDOFF-ENGINE-CLONE-REVIEW-2026-08-28.md).
> Blocker two (`state_cache.strides[0]`) is untouched by this and remains the gating item.

**Consequence for the test matrix:** any PP configuration must run with speculation
disabled. PP therefore cannot be compared like-for-like against the MTP-on production
baseline. The honest comparison is PP-without-MTP against TP2-without-MTP — which is
an argument for running the MTP-off matrix regardless of whether PP ever works.

---

## 3. Blocker two: the DSA compressor rejects the TP=1 tensor shape

With `SPEC_ARGS=` (speculation off) startup progressed through weight load and through
B12X selection, then died in warmup:

```
ValueError: Invalid state_cache.strides[0] on argument #0 when calling:
  __call__(state_cache: Tensor([n0, n1, 2048], float32), ...)
  expected to be divisible by 16
```

Path: `attention.py:366` → `compressor.py:372` →
`sparse_attn_compress_cutedsl.py:2103` → CUTLASS DSL FFI. This is the DeepSeek sparse
attention (DSA) compressor, not the MoE path. Only `Worker_PP0` — the rank holding the
first layers, where the compressor lives — hits it.

### Control: it is not the third node, and not the layer split

| Config | Nodes | Weights/node | B12X | Result |
|---|---:|---:|:---:|---|
| TP=1 / PP=3, split `14,15,14` | 3 | ~52 GB | loaded | `Invalid state_cache.strides[0]` |
| TP=1 / PP=2, split `22,21` | 2 | ~78 GB | loaded | **identical error** |

Reproducing at PP=2 and PP=3, with different splits and different node counts,
eliminates both. The remaining difference from working production is `TP=1`:
production runs `TP=2`, where the state-cache width is sharded across two ranks.

### The likely cheap fix, untested

`compressor.py:299` documents the layout as
`[num_blocks, block_size, kv_dim+score_dim]`, so `strides[0] = block_size * 2048`. The
deployment sets `--block-size 256`, giving 524288 — which *is* divisible by 16. So the
state-cache KV group is evidently not getting `block_size=256`, and **a `--block-size`
sweep is the obvious next experiment.** It is one flag.

---

## 4. What this does and does not establish

**Established:**
- B12X loads under PP=3 on three nodes. Verified live and in source.
- ~~MTP and PP are mutually exclusive in this runtime, by class hierarchy.~~
  **Corrected 2026-08-28:** the *error* was real but the *conclusion* over-reached. The
  class-hierarchy check asks the wrong question — the draft runs whole on the last rank
  and never needed `SupportsPP`. A three-patch exemption exists publicly (see the update
  note in §2); what remains established is only that **stock vLLM refuses the combination**.
- The stride failure is independent of node count and layer partition.

**Not established:**
- **Any PP performance figure.** Every PP run died before serving a token. There is no
  PP tok/s number in this PR, and anyone citing one is citing something that does not exist.
- **That TP=1 is definitively the cause.** It is the last variable standing after the
  PP=2 control, but the clean test (`TP=2 / PP=2`, 4 ranks) was not run — each node has
  one GPU, so it needs two ranks per GB10, and an earlier over-subscription wedged a
  node hard enough to require a power cycle.

---

## 5. Operational findings worth keeping

### A single-node config will wedge the box, not OOM cleanly

`TP=1 / PP=1` on one node asks a 121 GiB GB10 to hold a 156 GB checkpoint. The node did
not fail cleanly: ping stayed at 0.2 ms and port 22 kept accepting TCP on all five
interfaces, but sshd could never complete a banner exchange. Reaching it from a peer
over the CX-7 fabric failed identically, which is what rules out a LAN/WiFi explanation.
Docker's TCP API is not exposed, so there is no out-of-band recovery — it is a power
cycle.

**`nc` / `/dev/tcp` port probes will report a wedged Spark as healthy. Read the SSH
banner.**

### The overlay network does not survive a reboot

The `192.168.99.x/30` addressing used by the EP3 run is runtime-only, as PR #3 warned.
A reboot cleared it along with the `known_hosts` entries for those addresses. The
persistent fabric is `192.168.100.1↔.2` (sparkmain↔spark1) and `192.168.101.1↔.2`
(sparkmain↔spark-sep). Re-establish the overlay before any 3-node run, or use the
persistent addresses.

### `VLLM_PP_LAYER_PARTITION` must be unset, never empty

Compose's `environment:` block always materializes an unset variable as `""`, and vLLM
parses `""` as a partition list and dies with `Invalid partition string:`. Inject it
conditionally instead:

```yaml
${VLLM_PP_LAYER_PARTITION:+export VLLM_PP_LAYER_PARTITION=${VLLM_PP_LAYER_PARTITION};}
```

(For 43 layers over 3 ranks vLLM auto-computes `14,15,14` anyway — the variable is
belt-and-braces, not a requirement.)

---

## 6. Runtime parameterization

The deployment hardcoded `--tensor-parallel-size 2`, `--pipeline-parallel-size 1` and
`--nnodes 2` in the compose command, so no architecture could be tested without editing
it. It is now parameterized on all three nodes:

```yaml
--tensor-parallel-size ${TP_SIZE:-2}
--pipeline-parallel-size ${PP_SIZE:-1}
--nnodes ${NNODES:-2}
--moe-backend ${MOE_BACKEND:-flashinfer_b12x}
${ENABLE_EP:+--enable-expert-parallel}
${SPEC_ARGS---speculative-config "$${SPECULATIVE_CONFIG}"}
```

`SPEC_ARGS=` (empty) disables speculation; unset keeps it on. **The defaults render the
original production command byte-for-byte** — verified by diffing
`docker compose config` output against the untouched production env file, not assumed.

Production was restored and re-verified afterwards:

| | Before | After restore |
|---|---:|---:|
| decode tok/s (median of 3) | 49.12 | 48.23 |
| TTFT | 0.209 s | 0.154 s |
| KV tokens | 1,776,414 | 1,855,255 |

---

## 7. Suggested next experiments

1. **`--block-size` sweep at TP=1/PP=2.** One flag, and the only thing between this
   branch and an actual PP measurement.
2. **`TP=2 / PP=2`** — settles the TP=1 theory. Watch memory: two ranks per GB10.
3. **MTP-off matched matrix** (`TP2` / `EP2` / `EP3`), recording the selected MoE
   kernel and target steps/sec. Removes speculation as a confounder from PR #3's
   numbers. Independent of PP, runnable today.
4. **The official NVIDIA 3-node NCCL ring** (`--topology ring`,
   `NCCL_IB_SUBNET_AWARE_ROUTING=1`, `NCCL_NET_PLUGIN=none`) before concluding anything
   about needing a switch — neither variable appears in PR #3's falsified list.
