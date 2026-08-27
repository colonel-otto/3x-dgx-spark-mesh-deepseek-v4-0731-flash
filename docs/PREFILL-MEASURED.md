# Prefill investigation — retired

The original seven-addendum narrative is retired because its intermediate explanations
were repeatedly falsified. Its final result is that a silent RDMA degradation on `spark1`,
not a TP=3 configuration limit, caused the apparent prefill gap. Rebooting the node
restored 8K and 32K prefill to 95–96% of the upstream reference.

If you arrived here for the old “two prefill rates, ~30x apart” claim: it was false. The
high server-log values were cache-assisted 10-second windows, not sustained kernel rates.
If you arrived from the single-node comparison: the 155.43 GiB checkpoint still does not
fit safely on one approximately 121 GiB Spark; that capacity fact was unrelated to the
prefill diagnosis.

- Current resolved finding: [fabric-fix parity](FABRIC-FIX-PARITY.md).
- Current fabric interpretation: [bandwidth comparison](BANDWIDTH-COMPARISON.md).
- Failure signature and recovery: [degraded-data catalogue](DEGRADED-DATA-CATALOGUE.md)
  and [troubleshooting](troubleshooting.md).
- Current configuration rationale: [decisions](DECISIONS.md).
- Frozen evidence: [`results/20260824-prefill/`](../results/20260824-prefill/) and
  [`results/20260825-fabric-fix/`](../results/20260825-fabric-fix/).

[View the complete retired investigation at its last full revision.](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/PREFILL-MEASURED.md)
