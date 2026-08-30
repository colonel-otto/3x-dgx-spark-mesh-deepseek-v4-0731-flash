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

## The re-measure (ready to run, engine currently DOWN)

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
