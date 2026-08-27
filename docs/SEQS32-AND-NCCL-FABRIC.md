# Sequence-cap and NCCL investigation — retired

The 2026-08-24 rejection of `MAX_NUM_SEQS=32` was taken on degraded fabric and is
overturned. On healthy fabric, `seqs=32` improved aggregate throughput at concurrency 32
with 70/70 successful requests; it is the current production value.

What survives is the diagnostic signature: matching `_ALLGATHER_BASE` watchdog timeouts
on every rank, with no KV pressure, indicate a collective that cannot complete within its
budget. Gate the fabric before lowering the sequence cap or drawing a hardware conclusion.

- Current value and rationale: [decisions](DECISIONS.md).
- Symptom, confirmation, and recovery: [degraded-data catalogue](DEGRADED-DATA-CATALOGUE.md)
  and [troubleshooting](troubleshooting.md).
- Official bandwidth result: [bandwidth comparison](BANDWIDTH-COMPARISON.md).
- Frozen original and retest evidence:
  [`20260824-seqs32-nccl`](../results/20260824-seqs32-nccl/) and
  [`20260826-seqs32-retest`](../results/20260826-seqs32-retest/).

The complete investigation remains available through Git history.
