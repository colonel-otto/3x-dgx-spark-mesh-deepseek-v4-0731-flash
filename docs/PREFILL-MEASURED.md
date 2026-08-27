# Prefill investigation — retired

The original seven-addendum narrative is retired because its intermediate explanations
were repeatedly falsified. Its final result is that a silent RDMA degradation on `spark1`,
not a TP=3 configuration limit, caused the apparent prefill gap. Rebooting the node
restored 8K and 32K prefill to 95–96% of the upstream reference.

- Current resolved finding: [fabric-fix parity](FABRIC-FIX-PARITY.md).
- Current fabric interpretation: [bandwidth comparison](BANDWIDTH-COMPARISON.md).
- Failure signature and recovery: [degraded-data catalogue](DEGRADED-DATA-CATALOGUE.md)
  and [troubleshooting](troubleshooting.md).
- Current configuration rationale: [decisions](DECISIONS.md).
- Frozen evidence: [`results/20260824-prefill/`](../results/20260824-prefill/) and
  [`results/20260825-fabric-fix/`](../results/20260825-fabric-fix/).

The complete investigation remains available through Git history.
