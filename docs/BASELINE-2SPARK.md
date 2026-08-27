# 2-Spark baseline — retired

This 2026-08-21 baseline traversed the link later found to have silent RDMA degradation.
Its timing values are diagnostic signatures, not a valid 2-node comparison. Its
transport-independent KV accounting and correctness observations are retained in the raw
bundle.

If you arrived from the experiment log for the `48.23 tok/s` reference: that timing is
void. It crossed the degraded `sparkmain`–`spark1` link and must not be compared with a
healthy run.

For current comparison status, use the [dated handoff](HANDOFF-2026-08-27.md),
[decisions](DECISIONS.md), and [results index](../results/INDEX.md). For the symptom and
fabric gate, use the [degraded-data catalogue](DEGRADED-DATA-CATALOGUE.md). Raw evidence:
[`20260821T001024Z-2spark-baseline`](../results/20260821T001024Z-2spark-baseline/).

[View the complete retired baseline at its last full revision.](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/BASELINE-2SPARK.md)
