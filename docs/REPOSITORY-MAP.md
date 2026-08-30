# Repository and data map

This page answers two questions: where each kind of document or benchmark belongs, and
which material in the surrounding workstation should remain local.

The canonical GitHub repository is
[`colonel-otto/3x-dgx-spark-mesh-deepseek-v4-0731-flash`](https://github.com/colonel-otto/3x-dgx-spark-mesh-deepseek-v4-0731-flash). Only the
`3spark-dsv4/` checkout is repository content. Sibling checkouts and the workspace-level
`docs/`, `configs/`, and scripts are not staging folders for this repository.

## What belongs in the repository

| Material | Destination | Requirement |
|---|---|---|
| Reproduction setup, topology, patches, and generic operations | `docs/` | No host secrets; commands must be reusable by another operator |
| A benchmark or quality run | `results/YYYYMMDD-<subject>/` | README, live config capture, harness/version, raw reps, and fabric gate when performance is measured |
| Normalized observations | `benchmarks/measurements.csv` | Prompt, harness, statistic, source, and comparability recorded |
| Stable example configuration | `configs/*.env.example` or `config/` | Placeholder addresses and no credentials |
| Reusable automation | `scripts/` | Parameterized; no workstation-specific paths or embedded hosts |
| A settled finding or failure mode | `docs/DECISIONS.md`, a dated report, or troubleshooting docs | Link directly to committed evidence |
| A run on a **third-party** harness | `results/YYYYMMDD-<subject>/` plus a dated report | Name the tool and its exact version; state which comparisons are valid across harnesses and which are not (see [`RESULT-LLAMA-BENCHY-2V3-2026-08-30.md`](RESULT-LLAMA-BENCHY-2V3-2026-08-30.md)) |

Raw evidence belongs in Git when it is small enough to review and necessary to reproduce
a published claim. Prefer text, CSV, JSON, or JSONL. Do not commit only a screenshot or a
median when the per-rep records exist.

## What stays local

| Local material | Why it stays local | Repository-safe form, if useful |
|---|---|---|
| `../hosts.json` and workspace `configs/` | LAN inventory, real addresses, and live service wiring | Sanitized examples with placeholder hosts |
| `configs/2spark-live.env`, `configs/3spark-live.env`, and other live `.env` files | Deployment-specific values and possible secrets | Existing `*.env.example` files |
| SSH config, credentials, tokens, private keys, and unredacted environment captures | Secret or identity-bearing | Never commit; document variable names only |
| `.claude/`, caches, `__pycache__/`, editor state, and temporary outputs | Tool state, not project evidence | None |
| `../3spark-dsv4-pr6/` | Old worktree/checkpoint; not canonical | Reconcile intentional commits through Git, not file copying |
| `../MiaAI-Lab-2spark/` | Upstream/reference checkout | Cite upstream or vendor a specific file with provenance when necessary |
| Host recovery, DNS, desktop, and unrelated LAN service notes in `../docs/` | Useful private operations but outside this repo's reproducible DGX benchmark scope | Distill only the DGX-generic procedure after redaction |

## Workspace documentation audit

The surrounding `../docs/` directory contains three kinds of material:

| Local document group | Disposition |
|---|---|
| `dsv4-2spark-baseline.md`, old `dgx spark/` result reports, and old verdicts | **Keep local as archive.** The repository has frozen evidence and newer provenance labels; copying these would reintroduce superseded claims. |
| `dsv4-repetition-loop.md`, `dsv4-tuning.md`, and `dsv4-2node-fallback.md` | **Already represented or superseded.** Use repository pages such as `REPETITION-LOOP.md`, `DECISIONS.md`, and the current handoff. Reconcile only a specific missing fact. |
| `dsv4-restart.md` and `dgxsparkmain.md` | **Candidate for selective migration.** Extract generic restart/rollback steps, remove real paths, hosts, addresses, and live service details, then update the current handoff or troubleshooting guide. |
| `spark-gpu-wedge.md` | **Candidate for a separate sanitized troubleshooting note** if the failure is reproducible and DGX-generic; it is not benchmark evidence. |
| `dns-setup.md`, `spark-sep-recovery.md`, `spark-sep-chatbot-plan.md`, and workspace `topology.md` | **Local-only.** These describe the wider LAN, physical recovery, or other services. |

Do not bulk-copy the workspace documentation into the repo. Several local pages predate
the fabric fix and the output-window discovery, so their performance claims are useful as
history but unsafe as current guidance.

## Publication checklist

Before moving a local artifact into Git:

1. Confirm it supports this repository's three-Spark deployment or benchmark scope.
2. Remove credentials, real management addresses, MACs, serials, usernames, and absolute
   home paths; run `make check-sensitive`.
3. Separate narrative from evidence: narrative goes in `docs/`, raw runs in `results/`,
   normalized observations in `benchmarks/`.
4. Give a result a date, status, node/TP shape, live config source, harness/version, actual
   output-token count, rep count, statistic, fabric-gate status, and raw-file links.
5. Add or update its entry in `results/index.yaml` and `results/README.md`.
6. If a newer run reverses an older claim, keep the old bundle and mark it `VOID` or
   `SUPERSEDED`; never silently rewrite the decision trail.

The [benchmark policy](BENCHMARK-POLICY.md) defines the evidence bar. The
[provenance index](../results/INDEX.md) shows how existing bundles meet—or fail—that bar.

## Iterative documentation review

Run this loop after a benchmark changes a conclusion and before a documentation-focused
pull request:

1. **Inventory:** run `py scripts/audit_docs.py` (or `python3` on Linux). Review the
   largest pages, pages with no inbound links, broken local links, and stale-status terms.
2. **Group:** assign every page one job: landing/index, living how-to, policy, focused
   explanation, decision log, or retired redirect. A page with two jobs is a merge
   candidate.
3. **Choose an owner:** for each question, name one canonical page. Other pages link to
   it instead of restating its answer.
4. **Compare:** distinguish exact duplication from repeated interpretation. Exact text is
   easy to remove; repeated conclusions are more dangerous because their status drifts.
5. **Decide:** mark each candidate `KEEP`, `MERGE`, `RETIRE`, or `DELETE`.
6. **Reduce:** merge unique material into the owner. Leave a short redirect when frozen
   result bundles link to the old path; otherwise delete the obsolete page. Git history is
   the archive.
7. **Verify:** run the link audit, tests, schema check, and sensitive-data scan. Record the
   before/after page and word counts in the pull request.

### Canonical owner by question

| Question | Owner |
|---|---|
| What should a new reader believe now? | Root `README.md` |
| Do two or three nodes serve this model faster? | `RESULT-2V3-MATCHED-2026-08-30.md`, with `RESULT-LLAMA-BENCHY-2V3-2026-08-30.md` as the independent third-party corroboration |
| What is running and what remains open? | `HANDOFF-2026-08-30-ENGINE-AB.md` (latest dated handoff) |
| How do I build and start it? | `setup.md`, with `topology.md` and `patch.md` as focused dependencies |
| How must benchmarks be run? | `BENCHMARK-POLICY.md` |
| What is the status of a run? | `results/index.yaml`, rendered concisely in `results/INDEX.md` |
| Why was a configuration chosen? | `DECISIONS.md` |
| What happened chronologically? | `EXPERIMENT-LOG.md` |
| What does a known failure look like? | `DEGRADED-DATA-CATALOGUE.md` and `troubleshooting.md` |

Focused reports may explain one result in depth, but they must not maintain a second
project-wide scoreboard or current-state summary.

### Review queue after iteration 1

| Pages | Decision | Reason |
|---|---|---|
| `HANDOFF.md`, `WHY-THREE-NODES.md`, `BANDWIDTH-NEXT-TEST.md`, `results.md` | **RETIRE completed** | They repeated current state or conclusions that were later reversed. Short redirects preserve old links; Git preserves the full text. |
| `ACCEPTANCE.md`, `IMPLEMENTATION.md` | **DELETE completed** | Early project plans with no unique current instruction and no result-bundle dependencies. Setup and policy own this material. |
| `DECISIONS.md`, `setup.md`, `BENCHMARK-METHODOLOGY.md` | **FIX completed** | Void conclusions and retired-profile guidance no longer appear as current authority. The methodology page is now a compatibility redirect to policy and benchmark schema. |
| `PREFILL-MEASURED.md` | **MERGE completed** | Its surviving result is owned by `FABRIC-FIX-PARITY.md`, `BANDWIDTH-COMPARISON.md`, and troubleshooting; the former page is a compact redirect to those owners and the raw bundles. |
| `SEQS32-AND-NCCL-FABRIC.md` | **MERGE completed** | The current sequence-cap decision is in `DECISIONS.md`; the failure signature is in the degraded-data catalogue; the former page is a compact redirect to both result bundles. |
| `TP3-TUNING.md`, `BASELINE-2SPARK.md`, `KV-QUALITY-LONG-CONTEXT.md`, `MTP5-1M-AND-UPSTREAM-COMPARISON.md` | **RETIRE completed** | Compact redirects retain frozen links while current decisions and raw results own the surviving evidence. |
| `reproduction-methodology.md`, `NCCL-TESTS-BUILD.md`, `TTFT-AND-WARMUP.md` | **MERGE completed** | Reusable protocol is now owned by benchmark policy and setup; former pages remain as short compatibility redirects. |
| `EP3-EXPERT-PARALLEL.md`, `PP3-PIPELINE-PARALLEL.md` | **KEEP, then trim** | They contain unique source-level blockers, but operational discoveries duplicated elsewhere can move to troubleshooting. |
| `POSTMORTEM-2026-08-25.md`, `DEGRADED-DATA-CATALOGUE.md` | **KEEP** | They overlap in topic but serve different jobs: causal narrative versus symptom lookup. |
