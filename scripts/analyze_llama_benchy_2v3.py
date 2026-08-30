#!/usr/bin/env python3
"""Compare the llama-benchy TP=2 and TP=3 arms and judge the pre-registered
expectations in docs/PLAN-LLAMA-BENCHY-2V3.md.

WHAT THIS MAY AND MAY NOT CONCLUDE
----------------------------------
llama-benchy's `--depth N` prefills N tokens of *cached context* and measures on
top of it. Our own decode_depth_sweep.py sends an N-token prompt with *no*
caching. These are different measurements, so this script NEVER compares an
absolute t/s figure across harnesses. It compares only the 2v3 *ratio* computed
within llama-benchy, against the 2v3 ratio computed within our harness.

That is expectation L2, and it is load-bearing.

SIGNIFICANCE
------------
llama-benchy reports mean and std per cell over --runs repetitions. A ratio is
only reported as a direction when the two arms' means are separated by more than
the pooled spread; otherwise the cell is INCONCLUSIVE at this n. The plan says
to raise n on a disagreeing cell and re-run rather than report a marginal
difference, so a wide cell is flagged, not spun.

Usage:
  analyze_llama_benchy_2v3.py RUN_DIR [--json OUT.json]
"""
import argparse
import json
import math
import pathlib
import sys

# Our own harness's measured 2v3 decode advantage, from
# RESULT-2V3-MATCHED-2026-08-30.md. L3 asks whether llama-benchy's ratio lands
# within ~5 percentage points of this band.
OURS_LOW, OURS_HIGH = 17.0, 20.0
L3_TOLERANCE_PP = 5.0


def load_arm(run_dir: pathlib.Path, arm: str, kind: str):
    """Return {cell_key: {...}} for one arm/sweep, or None if absent."""
    path = run_dir / arm / f"{arm}-{kind}.json"
    if not path.exists():
        return None
    with path.open() as fh:
        doc = json.load(fh)
    cells = {}
    for b in doc.get("benchmarks", []):
        key = (b.get("concurrency", 1), b.get("context_size", 0),
               b.get("prompt_size", 0), b.get("response_size", 0))
        cells[key] = b
    return {"doc": doc, "cells": cells}


def stat(cell, metric):
    """(mean, std, n) for a metric, or None when the cell lacks it."""
    m = cell.get(metric)
    if not isinstance(m, dict) or m.get("mean") is None:
        return None
    vals = m.get("values") or []
    return float(m["mean"]), float(m.get("std") or 0.0), len(vals)


def welch(m2, s2, n2, m3, s3, n3):
    """Welch's t on the two arms. Returns (t, approx_significant).

    Exact p-values need scipy, which is not installed on the box. |t| >= 2 is
    used as the reporting threshold, which is about p<0.05 for these n.
    """
    if n2 < 2 or n3 < 2:
        return None, False
    se = math.sqrt((s2 * s2) / n2 + (s3 * s3) / n3)
    if se == 0:
        return None, False
    t = (m3 - m2) / se
    return t, abs(t) >= 2.0


def fmt_cell(key):
    cc, depth, pp, tg = key
    return f"cc={cc} depth={depth} pp={pp} tg={tg}"


def compare(run_dir, kind, metric, label, out):
    a2 = load_arm(run_dir, "tp2", kind)
    a3 = load_arm(run_dir, "tp3", kind)
    if not a2 or not a3:
        missing = [a for a, v in (("tp2", a2), ("tp3", a3)) if not v]
        print(f"\n## {label}\n  MISSING ARM(S): {', '.join(missing)} -- cannot compare")
        return

    print(f"\n## {label}  ({metric})")
    print(f"{'cell':<34} {'2-node':>12} {'3-node':>12} {'3v2':>9}  verdict")
    print("-" * 88)

    rows = []
    for key in sorted(set(a2["cells"]) & set(a3["cells"])):
        s2 = stat(a2["cells"][key], metric)
        s3 = stat(a3["cells"][key], metric)
        if not s2 or not s3:
            continue
        m2, sd2, n2 = s2
        m3, sd3, n3 = s3
        if m2 == 0:
            continue
        pct = (m3 - m2) / m2 * 100.0
        t, sig = welch(m2, sd2, n2, m3, sd3, n3)

        if not sig:
            verdict = "INCONCLUSIVE (raise n)"
        elif pct > 0:
            verdict = "3 nodes faster"
        else:
            verdict = "2 NODES FASTER <-- contradicts L1"

        print(f"{fmt_cell(key):<34} {m2:>8.2f}±{sd2:<3.0f} {m3:>8.2f}±{sd3:<3.0f} "
              f"{pct:>+8.1f}%  {verdict}")
        rows.append({
            "cell": {"concurrency": key[0], "depth": key[1],
                     "pp": key[2], "tg": key[3]},
            "tp2": {"mean": m2, "std": sd2, "n": n2},
            "tp3": {"mean": m3, "std": sd3, "n": n3},
            "pct_3_over_2": pct,
            "welch_t": t,
            "significant": sig,
            "verdict": verdict,
        })
    out[f"{kind}_{metric}"] = rows
    return rows


def judge(out):
    """Evaluate L1 and L3 against everything measured."""
    print("\n" + "=" * 88)
    print("PRE-REGISTERED EXPECTATIONS (docs/PLAN-LLAMA-BENCHY-2V3.md §5)")
    print("=" * 88)

    tg_rows = [r for k, rows in out.items() if k.endswith("tg_throughput")
               for r in rows]
    if not tg_rows:
        print("  no decode cells -- nothing to judge")
        return

    sig = [r for r in tg_rows if r["significant"]]
    contradict = [r for r in sig if r["pct_3_over_2"] < 0]
    incon = [r for r in tg_rows if not r["significant"]]

    # L1: direction. Three nodes faster in every cell that resolves.
    if contradict:
        print(f"  L1 FALSIFIED: {len(contradict)} cell(s) show TWO nodes faster "
              f"beyond their own spread:")
        for r in contradict:
            c = r["cell"]
            print(f"    cc={c['concurrency']} depth={c['depth']}: "
                  f"{r['pct_3_over_2']:+.1f}%")
    elif not sig:
        print("  L1 UNRESOLVED: no cell separated beyond its own spread at this n.")
    else:
        print(f"  L1 HELD: three nodes faster in all {len(sig)} resolved cell(s).")
    if incon:
        print(f"       ({len(incon)} cell(s) inconclusive -- plan says raise n and re-run)")

    # L3: magnitude. Does the ratio land within tolerance of our harness's band?
    if sig:
        pcts = [r["pct_3_over_2"] for r in sig]
        lo, hi = min(pcts), max(pcts)
        mean = sum(pcts) / len(pcts)
        print(f"\n  L3: llama-benchy 2v3 decode advantage "
              f"{lo:+.1f}%..{hi:+.1f}% (mean {mean:+.1f}%)")
        print(f"      our harness: +{OURS_LOW:.0f}%..+{OURS_HIGH:.0f}%, "
              f"tolerance ±{L3_TOLERANCE_PP:.0f} pp")
        if mean < OURS_LOW - L3_TOLERANCE_PP or mean > OURS_HIGH + L3_TOLERANCE_PP:
            print("      L3 FALSIFIED -- the two harnesses measure different things.")
            print("      That is a FINDING, not a failure: investigate before publishing.")
        else:
            print("      L3 HELD -- the ratio corroborates our harness.")

    print("\n  L2 is not testable and needs no test: absolute t/s are EXPECTED to")
    print("  differ (different corpus, different depth semantics). This script")
    print("  never compares absolutes across harnesses.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run_dir", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    if not args.run_dir.is_dir():
        sys.exit(f"no such run dir: {args.run_dir}")

    print(f"# llama-benchy 2v3 -- {args.run_dir.name}")
    print("\nCross-harness ABSOLUTE numbers are not comparable (L2). Only the")
    print("2v3 ratio computed within this harness may be read against ours.")

    out = {}
    # Decode is the headline claim; prefill is reported for completeness.
    compare(args.run_dir, "depth", "tg_throughput", "Depth sweep -- decode", out)
    compare(args.run_dir, "depth", "pp_throughput", "Depth sweep -- prefill", out)
    compare(args.run_dir, "concurrency", "tg_throughput",
            "Concurrency sweep -- decode (aggregate)", out)
    compare(args.run_dir, "concurrency", "tg_req_throughput",
            "Concurrency sweep -- decode (per-request)", out)

    judge(out)

    if args.json:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
