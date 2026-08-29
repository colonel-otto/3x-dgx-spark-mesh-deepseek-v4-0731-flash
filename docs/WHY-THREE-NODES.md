# Retired comparison narrative — why three nodes

This page's former 2-vs-3 recommendation depended on decode runs with collapsed output
windows. Both the magnitudes and the shape of that comparison were withdrawn. A corrected
three-node arm exists, but the matching corrected two-node arm does not.

Use these sources instead:

- [3-Node vs 2-Node Benchmark](BENCHMARK-2V3-NODES.md) — the complete, audited performance matrix and architectural analysis.
- [Root README](../README.md#current-evidence) — the current answer readers may rely on.
- [Current handoff](HANDOFF-2026-08-28.md) — production cluster configuration and tuning conclusions.
- [Corrected 3-node run](../results/20260827-decode-3node-fixed/) — valid single-arm data.
- [Provenance index](../results/INDEX.md) — status of every older result.

This path remains as a redirect because frozen result bundles cite it. The retired
narrative can be
[viewed on GitHub at its last full revision](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/WHY-THREE-NODES.md):

```bash
git show 78a91e1:docs/WHY-THREE-NODES.md
```
