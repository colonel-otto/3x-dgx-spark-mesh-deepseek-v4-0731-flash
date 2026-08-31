# Retired handoff — 2026-08-25

This handoff was retired because it mixed operating instructions with provisional
benchmarks and several conclusions that were later reversed. Keeping it as a second
current-state page made the repository harder to trust.

Use these canonical pages instead:

- **[Current handoff](HANDOFF-2026-08-31-MATCHED-ENGINE-AB.md) — START HERE.** The matched
  cross-engine A/B is done and the engine question is settled: **eugr wins every aggregate
  cell by +31 % to +61 % and single-stream decode by +38 %**, with anemll keeping 1.86× the
  KV pool (permanent) and per-stream decode at the c=16 cap. The earlier table *understated*
  eugr — it compared against 10-day-old anemll rows and its single-stream deltas sat inside
  the 12 % parity tolerance.
- [K-sweep handoff](HANDOFF-2026-08-30-EVENING-KSWEEP.md) — still current for everything
  except the cross-engine table: eugr is the serving engine (`:8100`), tuning settled at
  `nst=5`/`mnbt=8192`, gateway live, K sweep done.
- [Engine-A/B handoff](HANDOFF-2026-08-30-ENGINE-AB.md) — superseded. Records the engine
  switch, but its arm-1 throughput numbers were measured with kernel caches disabled and
  are lower bounds; its "c=16 scheduling cliff" is retracted.
- [Previous handoff](HANDOFF-2026-08-29-EVENING.md) — Live cluster state
  (including a `systemctl`-says-failed / engine-is-healthy trap), the significance finding
  that withdraws four of five published 2v3 decode deltas, five harness bugs fixed, and the
  exact next command.
- [Previous handoff](HANDOFF-2026-08-27.md) — superseded by the above for 2v3 conclusions.
- [Setup](setup.md) — build, start, verify, and switch cluster shapes.
- [Benchmark policy](BENCHMARK-POLICY.md) — required measurement gates.
- [Decisions](DECISIONS.md) — settled configuration choices.
- [Experiment log](EXPERIMENT-LOG.md) — chronological history.

Old result bundles still link here, so this redirect remains stable. The complete retired
text is preserved in Git history and can be
[viewed on GitHub at its last full revision](https://github.com/colonel-otto/DeepSeek-V4-Flash-3x-DGX-Spark/blob/78a91e1/docs/HANDOFF.md):

```bash
git show 78a91e1:docs/HANDOFF.md
```
