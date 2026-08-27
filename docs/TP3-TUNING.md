# TP=3 tuning sweep — retired

This 2026-08-21 sweep used the superseded `460800 / seqs=8 / 0.85 / MTP=4` profile on
degraded fabric. Its throughput magnitudes are not current evidence, and later matched
controls overturned its MTP and sequence-cap conclusions.

What survives: the TP=3 padding patch passed correctness and is required; the historical
throughput claim that TP=3 beat TP=2 is not current because the corrected matched 2-node
arm does not yet exist.

Current configuration choices belong in [decisions](DECISIONS.md); benchmark requirements
belong in [benchmark policy](BENCHMARK-POLICY.md). The degraded-fabric signature is in the
[degraded-data catalogue](DEGRADED-DATA-CATALOGUE.md). Frozen evidence remains under
[`results/`](../results/).

[View the complete retired sweep at its last full revision.](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/TP3-TUNING.md)
