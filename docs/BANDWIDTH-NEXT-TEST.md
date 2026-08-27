# Retired bandwidth test plan

This plan proposed experiments to explain an apparent bandwidth gap. The controlled test
later showed that the gap came from comparing different harnesses: official
`all_gather_perf` measured 23.92 GB/s on the same fabric.

Use these completed sources:

- [Bandwidth comparison](BANDWIDTH-COMPARISON.md) — conclusion and controlled variants.
- [Controlled raw bundle](../results/20260826-nccl-controlled/) — commands and raw logs.
- [Setup requirements](setup.md#2-configure-and-prove-the-ring) — version-matched reproduction.

This redirect preserves links from frozen results. The original test plan can be
[viewed on GitHub at its last full revision](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/BANDWIDTH-NEXT-TEST.md):

```bash
git show 78a91e1:docs/BANDWIDTH-NEXT-TEST.md
```
