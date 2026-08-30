# VOID — matched 2v3 attempt aborted mid-transition (2026-08-29)

**Status:** `VOID-operator-error`. Retained as a diagnostic baseline per
[`BENCHMARK-POLICY.md`](../../docs/BENCHMARK-POLICY.md) — *"nothing is deleted; it is
relabelled."*

**Do not cite any number in this directory.** The settled result is
[`20260830-matched-2v3-powered`](../20260830-matched-2v3-powered/) and
[`RESULT-2V3-MATCHED-2026-08-30.md`](../../docs/RESULT-2V3-MATCHED-2026-08-30.md).

## What this is

The TP=3 arm of the second matched attempt, at **n=7**. It completed all five depth cells
and the concurrency sweep, then the orchestrator died during the cluster transition to
TP=2, so **no TP=2 arm exists here**.

## Why it died — the lesson worth keeping

**The running script was overwritten by `scp` while bash was still executing it.** Bash
reads a script incrementally by byte offset; the file shifting underneath the interpreter
produced a syntax error at line 260 and the run aborted.

> **Never `scp` over a script that is currently running.** Write the replacement to a new
> path instead.

## What was salvaged

This arm's data was not wasted. It supplied:

- the **contamination audit** and the four-way proof that its `EXCLUSIVITY_FAIL` was a
  ledger bug rather than foreign traffic (analysis §4);
- the **clock envelope** (690 samples/node) establishing that GB10 never reaches 3003 MHz;
- the **deep-cell noise characterisation** — dropping one high and one low rep brings every
  cell to 5.7–8.5 %, which is what justified trusting medians at 131K and 262K;
- the **CVs** that the power analysis used to size n for the successful run.

Its n=7 TP=3 measurements are also preserved inside the successful bundle as
[`20260830-matched-2v3-powered/tp3-n7/`](../20260830-matched-2v3-powered/tp3-n7/), so the
n=7 and n=30 arms can be compared directly.
