# TP=3 attention-group padding

## Problem

The checkpoint uses eight output attention groups and 64 heads, or eight heads per
group. Tensor parallelism normally partitions those groups evenly. Eight does not
divide by three.

A naive integer division is unsafe: it can drop model structure while producing fluent
but incorrect output. Starting successfully is therefore not proof of a valid TP=3
deployment.

## Invariant-preserving approach

The tested patch changes the padded execution shape:

```text
output groups       8 -> 9
attention heads    64 -> 72
heads per group          8  (unchanged)
```

Nine groups divide evenly across three ranks. The extra group is padding and is removed
from the model-visible result. Preserving eight heads per group maintains the output
projection contract required by the checkpoint.

## Provenance and pinning

Implementation:
[`localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark`](https://github.com/localaiguyy/DeepSeek-V4-Flash-DSpark-3x-DGX-Spark)

Pinned publication revision:

```text
496c6a146a383f1b7c3f5991f4f1930091420720
```

Do not curl and execute the moving `main` branch. Clone it, check out the pinned commit,
review the patch, record its SHA-256 and apply it identically to every rank.

## Acceptance tests

The historical validation suite covered:

- deterministic capital lookup;
- `17 x 23 = 391` arithmetic;
- deterministic sentence completion;
- needle recall in an approximately 1,500-token prompt;
- a constrained red/blue reasoning problem (`red=7`, `blue=3`);
- degeneration detection using unique-word ratio;
- normal `finish_reason: stop` behavior.

The first red/blue run was incorrectly marked failed because the harness allowed only
96 output tokens and truncated chain-of-thought. At 400 tokens it produced the correct
answer. A correctness artifact must distinguish a wrong answer from output truncation.
