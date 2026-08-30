# Issue #22 exemption is STALE pending a 1M-depth re-measure (2026-08-30)

`benchmarks/CHANGELOG.md` ("`kv-cache-dtype=nvfp4_ds_mla` — kept") exempts us
from MiaAI-Lab issue #22 (`nvfp4_ds_mla` routed to the slow BF16 kernel,
collapsing long-context decode to ~1 tok/s at 600K+) on the strength of a sweep
that was flat 75–99 tok/s **up to 409,600 tokens** under the then-current
ceiling `max_model_len=460800`.

That exemption's boundary condition no longer holds:

- We now ship `MAX_MODEL_LEN=1048576` (settled: "1M context is FREE here").
- The suspect dispatch line is confirmed present, verbatim, in
  `dsv4-3spark:0.1.1` (verified 2026-08-30 in an ephemeral container):
  `v1/attention/backends/mla/flashmla_sparse.py:880` —
  `use_fp8_cache = self.kv_cache_dtype == "fp8_ds_mla"` — so `nvfp4_ds_mla`
  falls through to `_forward_bf16_kv`.
- MiaAI's symptom regime (600K+) is now inside our reachable envelope, and we
  have **zero decode measurements above 409,600 tokens**.

Status: **the exemption is STALE.** It is not evidence for or against the bug
between ~410K and 1M. Per the CHANGELOG's own warning, do not apply
`hotfix-nvfp4-ds-mla-issue22.sh` without re-measuring — and equally, do not
cite the old sweep as clearance for the new ceiling.

## The re-measure (RUNNING 2026-08-30 — partial results below)

`scripts/rerun_issue22_deep_decode.sh` — single-stream decode tok/s at depths
262K / 400K / 524K / 700K / 850K / ~1M, n>=5 per cell, asserted 256-token decode
windows, exclusivity-gated, results to a fresh `results/` bundle with README.
It refuses to run unless the head's `:8100` `/v1/models` endpoint answers
(default `BASE_URL=http://127.0.0.1:8100`; override from another box).

Decision rule when the numbers exist:

- Decode stays in family with the 262K cell through ~1M → exemption renewed at
  the new ceiling; record the bundle in the CHANGELOG section.
- Decode collapses toward ~1 tok/s anywhere in 524K–1M → issue #22 bites us;
  evaluate `../MiaAI-Lab-2spark/patches/hotfix-nvfp4-ds-mla-issue22.sh`
  (one-line dispatch fix, `NOT APPLIED` in our image as of 2026-08-30, byte-
  identical on MiaAI origin/main) with a matched before/after per
  `docs/BENCHMARK-POLICY.md`.

See also `docs/BACKPORT-DRYRUN-2026-08-30.md` for the in-container verification
transcript context.

## Live run 2026-08-30 — partial results and resume state

**Bundle:** `sparkmain:~/issue22-rerun/results/20260830T154448Z-issue22-deep-decode/`
(scripts staged LF-normalized at `sparkmain:~/issue22-rerun/scripts/`; log at
`sparkmain:/tmp/issue22-rerun.log`). Engine: clean 6-min cold start 11:37 EDT,
`GPU_MEMORY_UTILIZATION=0.82` (post-`59edcb6`), KV pool 30.99 GiB / 4,390,838
tokens, `max_model_len=1048576` verified live, exclusivity counter pinned at 0.
Note the 0.82 profile differs from published Profile B (0.835): decode absolutes
are comparable, but this bundle answers the COLLAPSE question, not tok/s parity.

| Depth | Status | Decode tok/s (n=7) | TTFT cold | Verdict |
|---:|---|---|---|---|
| 262,144 | **DONE** | median ~45.3 (35.7–49.2), all `ptok=257995`, `cached=0` | 182–221 s | **CLEAN** — matches published 45.0 baseline |
| 409,600 | running | warmup1 45.1 | — | pending (old exemption edge; expects flat) |
| 524,288 | queued | — | — | **first NEW cell** (danger zone starts here) |
| 716,800 | queued | — | — | pending resume |
| 870,400 | queued | — | — | pending resume |
| 1,046,528 | queued | — | — | pending resume |

**Early-stop decision (operator, 2026-08-30):** run through **524,288** and stop
if clean; the three deep cells are deferred. Rationale: 262K/410K only re-confirm
the old exemption; 524K is the first data point past it; the 700K–1M cells cost
~4–5 h of prefill wall-clock. **A clean 524K does NOT renew the exemption at 1M**
— MiaAI's collapse signature onset is 600K+, above the stop point. Until the deep
cells run, the honest claim is "measured clean through 524K; 600K–1M unmeasured."

**SCHEDULED: the deep cells run automatically at 02:00 EDT 2026-08-31** via a
one-shot systemd timer on sparkmain (`issue22-deep-cells.timer`, runs as user
sparkmain, log `/tmp/issue22-rerun-deep.log`). The script's own gates apply at
fire time: it refuses if the engine is down or the cluster is not idle, so a
stopped or busy cluster simply skips the run (check the log in the morning).
Cancel with `sudo systemctl stop issue22-deep-cells.timer`.

**To resume manually instead (engine up, cluster idle):**
```bash
ssh sparkmain
cd ~/issue22-rerun
DEPTHS="716800 870400 1046528" nohup bash scripts/rerun_issue22_deep_decode.sh   > /tmp/issue22-rerun-deep.log 2>&1 &
```
Then fold both bundles into one results/ entry per repo convention.
