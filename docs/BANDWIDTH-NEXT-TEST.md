# Retired bandwidth test plan

This plan proposed experiments to explain an apparent bandwidth gap. The controlled test
later showed that the gap came from comparing different harnesses: official
`all_gather_perf` measured 23.92 GB/s on the same fabric.

Use these completed sources:

- [Bandwidth comparison](BANDWIDTH-COMPARISON.md) — conclusion and controlled variants.
- [Controlled raw bundle](../results/20260826-nccl-controlled/) — commands and raw logs.
- [NCCL test build](NCCL-TESTS-BUILD.md) — version-matched reproduction.

This redirect preserves links from frozen results. The original test plan remains in Git:

```bash
git show 78a91e1:docs/BANDWIDTH-NEXT-TEST.md
```
