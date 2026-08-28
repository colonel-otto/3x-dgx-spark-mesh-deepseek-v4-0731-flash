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

## Provenance and Hermetic Build

The required patches are audited, version-controlled, and tracked directly in this repository under [`patches/`](../patches/):
- `apply_tp3_patch.py`: Pads 8 attention groups to 9 for sharding, trims after gather.
- `hotfix-dsv4-issue26-hybrid-swa-min.py`: Fixes sliding window attention minimum block sizing.
- `hotfix-dsv4-issue27-partial-prefill-concurrency.py`: Fixes concurrent partial prefill scheduling.

To apply and verify these patches hermetically without runtime monkey-patching:

```bash
docker build -f docker/Dockerfile.runtime -t dsv4-3spark:0.1.1 .
```

The build executes `apply_tp3_patch.py --check` at image build time to assert patch application.

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
