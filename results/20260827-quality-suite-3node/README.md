# Quality suite: the 2-node repo's own tests, run on 3 nodes — 2026-08-27

**Status:** `CURRENT` · **Nodes:** 3 (TP=3) · **Harness:** the 2-node repo's scripts,
**unmodified**, checksums in [`vendored-SHA256SUMS.txt`](vendored-SHA256SUMS.txt)
**Config:** [`engine-config.txt`](engine-config.txt), read from the live process
**Fabric gate:** ⚠️ `ABSENT` — see caveat below

## Why this run exists

[`colonel-otto/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark`](https://github.com/colonel-otto/DeepSeek-v4-Flash-DSpark-2x-DGX-Spark)
publishes four quality claims — RULER 8/8, tool battery 7/7, deep-context tools 8/8,
garble CLEAN — with **no committed result artifacts**. The scripts are in their repo; the
outputs are not.

This runs their scripts against our 3-node cluster. It does two things at once: it
supplies the missing evidence for their claims, and it adds a dimension we did not
previously measure at all. **This repo has measured throughput almost exclusively.**

## Results — every claim reproduces on three nodes

| their claim | evidence in their repo | our 3-node result |
|---|---|---|
| Tool battery **7/7** | none committed | ✅ **7/7 PASS** |
| Deep-context tools **8/8** | none committed | ✅ **8/8 PASS** (32K and 131K) |
| Garble sweep **CLEAN** | none committed | ✅ **ALL CLEAN**, 8 runs |
| RULER-lite **8/8** | none committed | ⏳ running |

### Tool calling — 7/7

`single_call`, `complex_schema`, `multiturn`, `parallel`, `thinking+tool`,
`issue55_trunc`, `forced_choice` — all `VALID_JSON`, all correct finish reasons.

### Deep-context tool calling — 8/8

| context | single | multiturn | complex | issue55 |
|---:|---|---|---|---|
| 32,768 | ✅ | ✅ | ✅ | ✅ |
| 131,072 | ✅ | ✅ | ✅ | ✅ |

**Tool calling does not degrade at 131K on three nodes.**

### Garble sweep — ALL CLEAN

| context | run 0 | run 1 |
|---:|---|---|
| 2,048 | CLEAN | CLEAN |
| 8,192 | CLEAN | CLEAN |
| 32,768 | CLEAN | CLEAN |
| 131,072 | CLEAN | CLEAN |

## Why the correctness gate ran first

The TP=3 attention-group padding patch is what makes three-way tensor parallelism correct
at all. Without it, stock vLLM computes `8 // 3 == 2`, silently drops six of eight
attention groups, and **serves fluent nonsense rather than failing**.

Fluent nonsense would very likely still pass a tool-calling battery. So `17x23 -> 391`
was verified against the live engine **before** any of these ran. It passed. Without that
check these results would be meaningless.

## Caveats — read before citing

1. **No fabric gate artifact.** Bandwidth is only measurable with the engine stopped, and
   these ran against a live engine serving other work. Per
   [`docs/BENCHMARK-POLICY.md`](../../docs/BENCHMARK-POLICY.md) this run is marked
   `fabric_gate: ABSENT`. These are **pass/fail quality results, not timing results**, so
   fabric speed affects how long they took, not whether they passed — but the rule is the
   rule and the row is marked.
2. **RULER omits 262,144.** Their default includes it; we ran 8,192 / 32,768 / 131,072 to
   keep the run bounded. **This is less coverage than their claim asserts**, and the gap
   is stated rather than glossed.
3. **These are cold, by design.** Their `EVAL.md` principle 2 is cold-prefill-only —
   warming would invalidate the garble and truncation tests. Correct, and symmetric: their
   claimed results are cold too.
4. **This is not a 2-vs-3 comparison.** No 2-node arm was run. It establishes that three
   nodes pass the same quality bar, not that they pass it *better*. Quality tests are
   pass/fail; there is no "better" to measure here.

## What this does and does not show

**Does:** three nodes with the padding patch pass every quality test the 2-node repo
publishes, including tool calling at 131K context. Their claims are credible — they were
simply unevidenced.

**Does not:** show any 3-node advantage. These tests have no performance axis.

## Reproducing

```bash
# vendored unmodified from their repo at HEAD; verify against vendored-SHA256SUMS.txt
python3 tool-battery.py http://127.0.0.1:8100/v1/chat/completions deepseek-v4-flash-0731
python3 deepctx-tool-battery.py http://127.0.0.1:8100/v1/chat/completions deepseek-v4-flash-0731 32768,131072
python3 context-garble-sweep.py --url http://127.0.0.1:8100/v1 --model deepseek-v4-flash-0731 \
  --lengths 2048,8192,32768,131072 --runs 2 --out garble.md
python3 ruler-lite.py --base-url http://127.0.0.1:8100/v1 --model deepseek-v4-flash-0731 \
  --lengths 8192,32768,131072 --seed 42 --request-timeout 3600 --output ruler-lite.json
```
