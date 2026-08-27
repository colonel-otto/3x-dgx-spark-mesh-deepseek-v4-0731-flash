# KV long-context quality report — retired

The 2026-08-24 single-arm NVFP4 run was clean through 464K prompt tokens, but it was not
a dtype comparison. The later matched quality A/B is the authoritative evidence: 23 of 24
cells were byte-identical and no material speed or memory difference was demonstrated.

- Current dtype decision: [decisions](DECISIONS.md).
- Current quality evidence: [`20260826-kv-dtype-ab`](../results/20260826-kv-dtype-ab/) and
  [`20260827-quality-suite-3node`](../results/20260827-quality-suite-3node/).
- Original single-arm evidence: [`20260824-kv-quality`](../results/20260824-kv-quality/).

The complete report remains available through Git history.
