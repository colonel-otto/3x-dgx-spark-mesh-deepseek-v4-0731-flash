# VOID — llama-benchy 2v3, first attempt, aborted on the fabric gate (2026-08-30)

**Status:** `VOID-no-measurement`. Retained per
[`BENCHMARK-POLICY.md`](../../docs/BENCHMARK-POLICY.md) — *"nothing is deleted; it is
relabelled."*

**No measurements were taken.** The completed run is
[`20260830T101053Z-llama-benchy-2v3`](../20260830T101053Z-llama-benchy-2v3/) and
[`RESULT-LLAMA-BENCHY-2V3-2026-08-30.md`](../../docs/RESULT-LLAMA-BENCHY-2V3-2026-08-30.md).

## Why it is kept: the gate did its job

```
[2026-08-30T06:07:32-04:00] === Step 1: TP=3 arm ===
[2026-08-30T06:07:33-04:00] WARN: fabric gate returned non-zero (see tp3/fabric-gate.log)
[2026-08-30T06:07:33-04:00] Restoring open-webui ...
```

The run stopped **before the first measured request**, which is Requirement 1 working as
designed: *"`run_experiment.sh` ... refuses to benchmark on gate failure."* Between
2026-08-21 and 2026-08-25 a node ran at ~15 % of its collective bandwidth with every error
counter reading zero and silently corrupted seven result directories. This directory is
what the correct outcome looks like instead — an aborted run and no data, rather than
plausible-looking numbers from a degraded fabric.

It also cleanly restored `open-webui` on the way out, so the abort left no side effects.

Contents are environment capture only: `orchestration.log`, `software-stack.txt`,
`tokenizer-check.txt`, and the gate artifacts under `tp3/`.
