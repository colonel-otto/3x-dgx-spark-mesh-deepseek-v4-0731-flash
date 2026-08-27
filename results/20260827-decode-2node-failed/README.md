# Failed corrected TP=2 decode attempt — 2026-08-27

**Status:** `VOID-incomplete-run` · **Measured samples:** 0 · **Fabric gate:**
`PRESENT-PASS` (15 pass, 0 fail, 2 expected engine skips; 8.87 GB/s)

This attempt is retained because the failure occurred after the TP=2 engine reported
ready and passed a live correctness request. Every subsequent benchmark request returned
HTTP 404; the final depth then encountered connection resets. All five JSONL files are
empty and no performance number can be derived from this bundle.

The immediately following attempt succeeded and is preserved in
[`../20260827-decode-2v3-fixed/`](../20260827-decode-2v3-fixed/). The transient failure was
not reproduced or root-caused. Keeping the logs prevents the successful retry from hiding
an operational failure mode.
