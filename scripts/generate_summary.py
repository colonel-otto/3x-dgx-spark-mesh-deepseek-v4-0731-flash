#!/usr/bin/env python3
"""Generate benchmarks/summary.csv from the two source-of-truth files.

summary.csv is DERIVED and must never be hand-edited. It exists so a reader can
see the headline results side by side WITHOUT being misled into comparing values
that are not comparable.

The two sources have different grains and must not be concatenated:

  historical-summary.csv  aggregated experiments (one row = a whole experiment,
                          carrying its own min/max/repetitions). Prompt is
                          `unrecorded` - these predate prompt attribution and the
                          prompt is NOT recoverable, so it is never inferred.

  measurements.csv        individual observations / sweep points (one row = one
                          measured point at one concurrency).

Concatenating them would make one aggregated experiment look equivalent to one
sweep point. Hence two files, and a generated summary that states comparability
explicitly per row:

  prompt-matched   same harness AND prompt_shape as the rows beside it; a direct
                   comparison is valid.
  historical-only  prompt unrecorded. Usable as a within-file trend only. Do NOT
                   divide against a prompt-matched value - on this deployment the
                   prompt alone moves single-stream decode by ~1.65x.
  external         published by a third party on their own hardware.
  capacity-metric  not a throughput number at all (e.g. KV cache tokens). Read
                   from the engine startup log and prompt-independent, so it is
                   comparable across configs without prompt matching.
"""
import csv
import io
import os

HERE = os.path.dirname(os.path.abspath(__file__))
BENCH = os.path.join(os.path.dirname(HERE), "benchmarks")

OUT_COLS = ["result_id", "source_file", "config_id", "metric", "statistic",
            "value", "prompt_shape", "harness", "comparability",
            "evidence_status", "notes"]


def read(name):
    with io.open(os.path.join(BENCH, name), encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def build():
    out = []

    # --- Legacy aggregated experiments: prompt unrecorded, never inferred. ---
    for r in read("historical-summary.csv"):
        out.append({
            "result_id": r["configuration"] + "-legacy",
            "source_file": "historical-summary.csv",
            "config_id": r["configuration"],
            "metric": "decode_tok_s",
            "statistic": "median",
            "value": r["decode_median_tok_s"],
            "prompt_shape": "unrecorded",
            "harness": "legacy-harness",
            "comparability": "historical-only",
            "evidence_status": r["evidence_status"],
            "notes": ("aggregated experiment; " + r["repetitions"] + " reps, range "
                      + r["decode_min_tok_s"] + "-" + r["decode_max_tok_s"]
                      + "; transport " + r["transport"]),
        })

    m = read("measurements.csv")

    def pick(config_id, harness, prompt_shape, concurrency, metric):
        for r in m:
            if (r["config_id"] == config_id and r["harness"] == harness
                    and r["prompt_shape"] == prompt_shape
                    and r["concurrency"] == concurrency
                    and r["reverted"] == "false" and r[metric]):
                return r
        return None

    # --- Headline single-stream decode, prompt-matched, per config. ---
    singles = [
        ("tp3-seqs8", "prompt-matched",
         "context-length sweep entry at a 256-token prompt"),
        ("tp3-seqs16", "prompt-matched",
         "8-rep noise floor study; observed range 66.6-88.5"),
        ("miaai-2spark-tp2", "external",
         "MiaAI-Lab published; 2x DGX Spark on their hardware"),
    ]
    for cfg, comp, note in singles:
        cands = [r for r in m
                 if r["config_id"] == cfg and r["harness"] == "bench-miaai"
                 and r["prompt_shape"] == "synthetic-numbered-words"
                 and r["concurrency"] == "1" and r["reverted"] == "false"
                 and r["decode_tok_s"]]
        if not cands:
            continue
        # For seqs16 prefer the later noise-floor row (median over 8 reps).
        r = cands[-1] if cfg == "tp3-seqs16" else cands[0]
        out.append({
            "result_id": cfg + "-c1-decode",
            "source_file": "measurements.csv",
            "config_id": cfg,
            "metric": "decode_tok_s",
            "statistic": r["statistic"],
            "value": r["decode_tok_s"],
            "prompt_shape": r["prompt_shape"],
            "harness": r["harness"],
            "comparability": comp,
            "evidence_status": "raw-measurements",
            "notes": note,
        })

    # --- Peak USEFUL aggregate: each config at its own max_num_seqs cap. ---
    for cfg, cc in (("tp3-seqs8", "8"), ("tp3-seqs16", "16")):
        r = pick(cfg, "bench-miaai", "synthetic-numbered-words", cc,
                 "aggregate_tok_s")
        if r:
            out.append({
                "result_id": cfg + "-peak-aggregate",
                "source_file": "measurements.csv",
                "config_id": cfg,
                "metric": "aggregate_tok_s",
                "statistic": r["statistic"],
                "value": r["aggregate_tok_s"],
                "prompt_shape": r["prompt_shape"],
                "harness": r["harness"],
                "comparability": "prompt-matched",
                "evidence_status": "raw-measurements",
                "notes": ("at the max_num_seqs cap (c=" + cc + "); peak USEFUL "
                          "concurrency. Higher c reads higher aggregate but that "
                          "is queueing, not capacity."),
            })

    # --- The prompt effect: same script, same engine, only the prompt differs. ---
    for ps, rid in (("code-brief", "prompt-effect-code"),
                    ("dense-prose", "prompt-effect-prose")):
        r = pick("tp3-seqs8", "ours-bench.py", ps, "1", "decode_tok_s")
        if r:
            out.append({
                "result_id": rid,
                "source_file": "measurements.csv",
                "config_id": "tp3-seqs8",
                "metric": "decode_tok_s",
                "statistic": r["statistic"],
                "value": r["decode_tok_s"],
                "prompt_shape": ps,
                "harness": "ours-bench.py",
                "comparability": "prompt-matched",
                "evidence_status": "raw-measurements",
                "notes": ("identical script and engine; this pair differs ONLY by "
                          "prompt - a 1.65x swing"),
            })

    # --- The MATCHED 2-node vs 3-node comparison (the headline result). ---
    # Same cluster, same afternoon, same harness and prompt shape, MTP=4 on both,
    # so node count is the only variable. Values are QUERIED from measurements.csv
    # rather than restated here, so the summary cannot drift from its source.

    # Aggregate at the shared c=16 sequence cap. 2-node wins this by ~19%.
    for cfg, rid in (("tp2-matched-mtp4", "tp2-matched-c16-aggregate"),
                     ("tp3-seqs16", "tp3-seqs16-c16-aggregate")):
        r = pick(cfg, "bench-miaai", "synthetic-numbered-words", "16",
                 "aggregate_tok_s")
        if r:
            out.append({
                "result_id": rid,
                "source_file": "measurements.csv",
                "config_id": cfg,
                "metric": "aggregate_tok_s",
                "statistic": r["statistic"],
                "value": r["aggregate_tok_s"],
                "prompt_shape": r["prompt_shape"],
                "harness": r["harness"],
                "comparability": "prompt-matched",
                "evidence_status": "raw-measurements",
                "notes": ("matched 2-vs-3-node comparison at the shared c=16 cap "
                          "(MTP=4 on both); 2-node leads by ~19%, a 1% trial "
                          "spread, so not noise"),
            })

    # Per-stream decode at 131K context, c=1. 3-node wins this.
    # The 3-node 131K point was measured on tp3-seqs8, not tp3-seqs16 - stated in
    # the note rather than silently relabelled.
    long_ctx = [
        ("tp2-matched-c1-decode-131k", ["tp2-matched-mtp4"]),
        ("tp3-seqs16-c1-decode-131k", ["tp3-seqs16", "tp3-seqs8"]),
    ]
    for rid, prefer in long_ctx:
        hit = None
        for cfg in prefer:
            for r in m:
                if (r["config_id"] == cfg and r["harness"] == "bench-miaai"
                        and r["prompt_shape"] == "synthetic-numbered-words"
                        and r["prompt_tokens"] == "131072"
                        and r["concurrency"] == "1"
                        and r["reverted"] == "false" and r["decode_tok_s"]):
                    hit = (cfg, r)
                    break
            if hit:
                break
        if not hit:
            continue
        cfg, r = hit
        note = ("matched 2-vs-3-node comparison; per-stream decode at 131,072-token "
                "context, c=1 (MTP=4 on both); 3-node leads by ~13% here and by "
                "8-17% across 2K-131K")
        if cfg != prefer[0]:
            note += (" | NOTE: not present under " + prefer[0] + "; sourced from "
                     "config_id " + cfg + ", which is the same 3-node TP=3 shape "
                     "differing only in max_num_seqs (8 vs 16) - a setting shown "
                     "not to move single-stream decode")
        out.append({
            "result_id": rid,
            "source_file": "measurements.csv",
            "config_id": cfg,
            "metric": "decode_tok_s",
            "statistic": r["statistic"],
            "value": r["decode_tok_s"],
            "prompt_shape": r["prompt_shape"],
            "harness": r["harness"],
            "comparability": "prompt-matched",
            "evidence_status": "raw-measurements",
            "notes": note,
        })

    # --- Deep concurrency: 4 x ~200,000-token prompts. ---
    # Both configs remain unusable at this depth, and preemptions stay 0 -- so
    # the KV pool genuinely never binds. But the ORIGINAL 2026-08-21 rows were
    # taken on the degraded spark1 fabric (issue #15), and the healthy-fabric
    # re-run is materially faster, so the surrounding "serialized prefill"
    # framing is scoped rather than restated as settled.
    #
    # Selection is by NEWEST matching run, and the note is derived from the row
    # rather than hardcoded -- an earlier version pinned prompt_tokens to exactly
    # "200000" and baked "~14.5 min wall" into the text, so a re-run would have
    # been skipped while the stale conclusion kept being asserted.
    for cfg, rid, label in (
            ("tp2-matched-mtp4", "tp2-matched-deep-concurrency", "2-node TP=2"),
            ("tp3-seqs16", "tp3-seqs16-deep-concurrency", "3-node TP=3")):
        cands = [c for c in m
                 if c["config_id"] == cfg and c["concurrency"] == "4"
                 and c["reverted"] == "false" and c["decode_tok_s"]
                 and c["prompt_tokens"].isdigit()
                 and 195000 <= int(c["prompt_tokens"]) <= 205000]
        if cands:
            r = max(cands, key=lambda c: c["timestamp_utc"])
            ttft = r["ttft_ms"] or "?"
            wall_min = ("%.1f" % (int(ttft) / 60000.0)) if ttft.isdigit() else "?"
            note = (label + ", 4 concurrent ~200,000-token prompts: still "
                    "UNUSABLE (TTFT " + ttft + " ms, i.e. >" + wall_min +
                    " min to first token). preemptions 0, so the KV pool never "
                    "binds at this depth. Measured " + r["timestamp_utc"][:10] +
                    " with harness " + r["harness"] + ".")
            if r["timestamp_utc"] < "2026-08-25":
                note += (" TAKEN ON THE DEGRADED spark1 FABRIC - provisional, "
                         "see issue #15.")
            out.append({
                "result_id": rid,
                "source_file": "measurements.csv",
                "config_id": cfg,
                "metric": "decode_tok_s",
                "statistic": r["statistic"],
                "value": r["decode_tok_s"],
                "prompt_shape": r["prompt_shape"],
                "harness": r["harness"],
                "comparability": "prompt-matched",
                "evidence_status": "raw-measurements",
                "notes": note,
            })

    # --- KV capacity: the structural 3-node result, prompt-independent. ---
    for cfg, label in (("ours-2spark-tp2-baseline", "2-node TP=2"),
                       ("tp3-seqs16", "3-node TP=3")):
        rs = [r for r in m if r["config_id"] == cfg and r["kv_cache_tokens"]]
        if rs:
            out.append({
                "result_id": cfg + "-kv-tokens",
                "source_file": "measurements.csv",
                "config_id": cfg,
                "metric": "kv_cache_tokens",
                "statistic": "engine-reported",
                "value": rs[-1]["kv_cache_tokens"],
                "prompt_shape": "n/a",
                "harness": "n/a",
                "comparability": "capacity-metric",
                "evidence_status": "engine-startup-log",
                "notes": label + "; KV capacity is prompt-independent",
            })
    return out


def main():
    rows = build()
    path = os.path.join(BENCH, "summary.csv")
    with io.open(path, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=OUT_COLS, lineterminator="\n")
        w.writeheader()
        w.writerows(rows)
    print("wrote " + path + " (" + str(len(rows)) + " rows)")


if __name__ == "__main__":
    main()
