#!/usr/bin/env python3
"""Schema and provenance checks for the benchmark CSVs.

The rule this file exists to enforce: on this deployment the benchmark prompt
alone moves single-stream decode by ~1.65x (81.8 vs 49.4 tok/s, same script and
engine). A throughput number without its prompt is therefore not a measurement.

New measurements MUST carry prompt attribution. `unrecorded` is permitted only
for legacy evidence in historical-summary.csv, where the prompt genuinely was
not recorded and must not be inferred after the fact.

Run:  python3 tests/test_benchmark_schemas.py
"""
import csv
import io
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "benchmarks")

MEASUREMENT_COLS = [
    "timestamp_utc", "config_id", "nodes", "tp_size", "pp_size", "max_model_len",
    "max_num_seqs", "mtp_num_tokens", "gpu_mem_util", "kv_cache_gib",
    "kv_cache_tokens", "max_concurrency_x", "observation_type", "statistic",
    "source", "reverted", "harness", "prompt_shape", "prompt_tokens",
    "concurrency", "decode_tok_s", "aggregate_tok_s", "ttft_ms",
    "accept_rate_pct", "accept_len", "notes",
]
SUMMARY_COLS = [
    "result_id", "source_file", "config_id", "metric", "statistic", "value",
    "prompt_shape", "harness", "comparability", "evidence_status", "notes",
]

# deepconc.py is a SEPARATE harness from bench-miaai, not a re-label of it. It
# reproduces bench-miaai's sampling and prompt shape but is a different script
# with a different TTFT definition (first SSE chunk) and its own cache-defeat
# and preemption instrumentation. The 2026-08-21 deep-concurrency run was ad-hoc
# and left no script, so its rows cannot be reproduced byte-for-byte -- calling
# both "bench-miaai" would hide exactly that.
# benchmark_prefill.py is upstream anemll's, run UNMODIFIED (it embeds a
# token_pool_sha256 that matched theirs byte-for-byte). It measures prefill
# server-side from token IDs, so its tok/s is a PREFILL rate and must never be
# divided against the decode rates the other harnesses produce.
VALID_HARNESS = {"bench-miaai", "benchmark_tp3", "ours-bench.py", "deepconc.py",
                 "benchmark_prefill.py"}
# random-token-ids: upstream's prefill harness feeds pseudo-random token IDs
# (seeded per size/trial so no two requests share a prefix). Not natural text at
# all, which is the point -- it defeats the prefix cache and makes prefill cost
# depend only on depth.
VALID_PROMPT = {"code-brief", "dense-prose", "synthetic-numbered-words",
                "random-token-ids"}
VALID_STATISTIC = {"median", "single-observation", "mean", "engine-reported", ""}
VALID_OBSERVATION = {"sweep-point", "acceptance-observation", "correctness-check"}
VALID_SOURCE = {"local-measurement", "external-published"}
VALID_COMPARABILITY = {"prompt-matched", "historical-only", "external",
                       "capacity-metric"}
THROUGHPUT_METRICS = ("decode_tok_s", "aggregate_tok_s")

failures = []


def check(cond, msg):
    if not cond:
        failures.append(msg)


def read(name):
    with io.open(os.path.join(BENCH, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def test_measurements():
    rows = read("measurements.csv")
    check(list(rows[0].keys()) == MEASUREMENT_COLS,
          "measurements.csv columns do not match the expected schema")

    for i, r in enumerate(rows, start=2):
        where = "measurements.csv line %d (%s)" % (i, r["config_id"])

        # THE core rule: any throughput value must carry a real prompt shape.
        has_throughput = any(r[m] for m in THROUGHPUT_METRICS)
        if has_throughput:
            check(r["prompt_shape"] in VALID_PROMPT,
                  "%s: throughput value without valid prompt_shape (got %r). "
                  "'unrecorded' is NOT allowed in measurements.csv."
                  % (where, r["prompt_shape"]))
            check(r["harness"] in VALID_HARNESS,
                  "%s: throughput value without valid harness (got %r)"
                  % (where, r["harness"]))
            check(r["statistic"] in VALID_STATISTIC and r["statistic"],
                  "%s: throughput value must state a statistic (got %r)"
                  % (where, r["statistic"]))

        check(r["prompt_shape"] != "unrecorded",
              "%s: 'unrecorded' is reserved for historical-summary.csv" % where)
        check(r["observation_type"] in VALID_OBSERVATION,
              "%s: bad observation_type %r" % (where, r["observation_type"]))
        check(r["source"] in VALID_SOURCE,
              "%s: bad source %r" % (where, r["source"]))
        check(r["reverted"] in ("true", "false"),
              "%s: reverted must be true/false (got %r)" % (where, r["reverted"]))
        check(r["statistic"] in VALID_STATISTIC,
              "%s: bad statistic %r" % (where, r["statistic"]))

        # External rows must be labelled, never silently mixed in.
        if r["config_id"].startswith("miaai-"):
            check(r["source"] == "external-published",
                  "%s: MiaAI row not marked external-published" % where)

    # Reverted experiments are preserved, not deleted.
    check(any(r["reverted"] == "true" for r in rows),
          "measurements.csv: reverted experiments must be PRESERVED and marked, "
          "not removed (expected the MAX_NUM_BATCHED_TOKENS=16384 rows)")


def test_historical():
    rows = read("historical-summary.csv")
    for i, r in enumerate(rows, start=2):
        where = "historical-summary.csv line %d (%s)" % (i, r["configuration"])
        # Legacy evidence: prompt is unrecoverable and must never be inferred.
        check(r["prompt_shape"] == "unrecorded",
              "%s: legacy rows must say 'unrecorded', never an inferred prompt "
              "(got %r)" % (where, r["prompt_shape"]))
        check(r["harness"] == "legacy-harness",
              "%s: legacy rows must say 'legacy-harness'" % where)
        check(r["observation_type"] == "aggregated-experiment",
              "%s: historical rows are aggregated experiments, not sweep points"
              % where)
        check(r["evidence_status"] == "historical-summary",
              "%s: bad evidence_status %r" % (where, r["evidence_status"]))


def test_summary_is_generated():
    rows = read("summary.csv")
    check(list(rows[0].keys()) == SUMMARY_COLS,
          "summary.csv columns do not match the expected schema")
    for i, r in enumerate(rows, start=2):
        where = "summary.csv line %d (%s)" % (i, r["result_id"])
        check(r["comparability"] in VALID_COMPARABILITY,
              "%s: bad comparability %r" % (where, r["comparability"]))
        check(r["source_file"] in ("measurements.csv", "historical-summary.csv"),
              "%s: every summarized result needs a source_file" % where)
        check(bool(r["result_id"]) and bool(r["config_id"]),
              "%s: every summarized result needs a stable result_id and config_id"
              % where)
        check(bool(r["statistic"]),
              "%s: every summarized result must state its statistic" % where)
        # Legacy values must not be dressed up as comparable.
        if r["prompt_shape"] == "unrecorded":
            check(r["comparability"] == "historical-only",
                  "%s: an unrecorded prompt can only be 'historical-only'" % where)

    ids = [r["result_id"] for r in rows]
    check(len(ids) == len(set(ids)), "summary.csv: result_id values must be unique")


def test_summary_matches_sources():
    """summary.csv is generated; regenerating must not change it."""
    path = os.path.join(BENCH, "summary.csv")
    with io.open(path, encoding="utf-8") as fh:
        before = fh.read()
    sys.path.insert(0, os.path.join(os.path.dirname(HERE), "scripts"))
    import generate_summary
    generate_summary.main()
    with io.open(path, encoding="utf-8") as fh:
        after = fh.read()
    check(before == after,
          "summary.csv is stale or was hand-edited. It is GENERATED - run "
          "`python3 scripts/generate_summary.py` and commit the result.")


def main():
    test_measurements()
    test_historical()
    test_summary_is_generated()
    test_summary_matches_sources()
    if failures:
        print("FAILED (%d)" % len(failures))
        for f in failures:
            print("  - " + f)
        return 1
    print("OK - benchmark schemas, prompt attribution, and provenance all valid")
    return 0


if __name__ == "__main__":
    sys.exit(main())
