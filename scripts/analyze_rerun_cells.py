#!/usr/bin/env python3
"""Compare the n=30 re-run of the two cells that did not resolve at n=10.

The re-run bundle's layout differs from the parent run's (one file per cell
rather than one per sweep), so `analyze_llama_benchy_2v3.py` cannot read it.

WHAT THIS REPORTS, AND WHY IT IS NOT JUST A SIGNIFICANCE TEST
------------------------------------------------------------
n=30 was pre-committed BEFORE these numbers existed. If a cell still does not
resolve, that is a legitimate outcome and is reported as such -- the confidence
interval still tightens, which is itself informative. Re-running a third time
hoping for significance would be p-hacking.

So this prints, for each cell:

  1. The 3v2 difference and its 95% CI.
  2. Whether the CI excludes the +15.4% depth-sweep effect (the question that
     actually matters: is the advantage SMALLER at this shape?).
  3. Whether the CI excludes zero (is there any advantage at all?).
  4. The variance change against the n=10 run, because the parent run's
     standard deviations turned out to be understated and that finding
     generalises beyond these two cells.

Usage:
  analyze_rerun_cells.py RERUN_DIR [--parent PARENT_DIR]
"""
import argparse
import json
import math
import pathlib
import sys

# The like-for-like figure from the parent run's depth sweep. The question for
# these two cells is whether their advantage is genuinely smaller than this.
DEPTH_SWEEP_EFFECT = 15.4


def load(path):
    if not path.exists():
        return None
    with path.open() as fh:
        return json.load(fh)


def first_cell(doc):
    """These files contain exactly one benchmark shape by construction."""
    if not doc or not doc.get("benchmarks"):
        return None
    return doc["benchmarks"][0]


def tg(cell):
    m = cell.get("tg_throughput") or {}
    if m.get("mean") is None:
        return None
    return float(m["mean"]), float(m.get("std") or 0.0), len(m.get("values") or [])


def compare(label, b2, b3, parent=None):
    s2, s3 = tg(b2), tg(b3)
    if not s2 or not s3:
        print(f"\n## {label}\n  missing data")
        return None

    m2, sd2, n2 = s2
    m3, sd3, n3 = s3
    diff = m3 - m2
    pct = diff / m2 * 100.0
    se = math.sqrt(sd2 * sd2 / n2 + sd3 * sd3 / n3)
    t = diff / se if se else float("nan")
    lo, hi = diff - 1.96 * se, diff + 1.96 * se
    lo_pct, hi_pct = lo / m2 * 100.0, hi / m2 * 100.0
    resolved = abs(t) >= 2.0

    print(f"\n## {label}")
    print(f"  2-node: {m2:6.2f} +- {sd2:5.2f}  (n={n2}, CV {sd2/m2*100:.1f}%)")
    print(f"  3-node: {m3:6.2f} +- {sd3:5.2f}  (n={n3}, CV {sd3/m3*100:.1f}%)")
    print(f"  3v2:    {pct:+.1f}%   Welch t = {t:.2f}")
    print(f"  95% CI: [{lo_pct:+.1f}%, {hi_pct:+.1f}%]")
    print()
    print(f"  resolves at |t|>=2?              {'YES' if resolved else 'NO'}")
    print(f"  CI excludes +{DEPTH_SWEEP_EFFECT}% (smaller here)? "
          f"{'YES' if hi_pct < DEPTH_SWEEP_EFFECT else 'NO'}")
    print(f"  CI excludes 0 (any advantage)?   "
          f"{'YES' if lo_pct > 0 else 'NO'}")

    if parent:
        p2, p3 = parent
        print()
        print("  variance vs the n=10 run (the parent run's spreads were understated):")
        for arm, new, old in (("2-node", s2, p2), ("3-node", s3, p3)):
            if not old:
                continue
            om, osd, on = old
            ratio = (new[1] / osd) if osd else float("nan")
            print(f"    {arm}: std {osd:.2f} (n={on}) -> {new[1]:.2f} (n={new[2]}), "
                  f"{ratio:.1f}x;  mean {om:.2f} -> {new[0]:.2f} "
                  f"({abs(new[0]-om)/om*100:.1f}% shift)")

    return {"label": label, "pct": pct, "t": t, "resolved": resolved,
            "ci_pct": [lo_pct, hi_pct],
            "excludes_depth_effect": hi_pct < DEPTH_SWEEP_EFFECT,
            "excludes_zero": lo_pct > 0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("rerun_dir", type=pathlib.Path)
    ap.add_argument("--parent", type=pathlib.Path)
    ap.add_argument("--json", type=pathlib.Path)
    args = ap.parse_args()

    d = args.rerun_dir
    if not d.is_dir():
        sys.exit(f"no such dir: {d}")

    print(f"# n=30 re-run of the inconclusive cells -- {d.name}")
    print("\nn=30 was pre-committed before these numbers existed. A cell that")
    print("still does not resolve is a legitimate outcome, not a reason to re-run.")

    # Parent-run values for the variance comparison, if available.
    parent = {"8k": None, "cc1": None}
    if args.parent and args.parent.is_dir():
        pd2 = load(args.parent / "tp2" / "tp2-depth.json")
        pd3 = load(args.parent / "tp3" / "tp3-depth.json")
        pc2 = load(args.parent / "tp2" / "tp2-concurrency.json")
        pc3 = load(args.parent / "tp3" / "tp3-concurrency.json")

        def pick(doc, key, val):
            if not doc:
                return None
            for b in doc["benchmarks"]:
                if b.get(key) == val:
                    return tg(b)
            return None

        parent["8k"] = (pick(pd2, "context_size", 8192), pick(pd3, "context_size", 8192))
        parent["cc1"] = (pick(pc2, "concurrency", 1), pick(pc3, "concurrency", 1))

    out = []
    for label, fname, pkey in (
        ("Cell A -- 8K decode (depth-sweep shape: pp=2048, depth=8192)",
         "depth8k", "8k"),
        ("Cell B -- cc=1 decode (concurrency-sweep shape: pp=8192, depth=0, --no-cache)",
         "cc1", "cc1"),
    ):
        b2 = first_cell(load(d / "tp2" / f"tp2-{fname}.json"))
        b3 = first_cell(load(d / "tp3" / f"tp3-{fname}.json"))
        if not b2 or not b3:
            print(f"\n## {label}\n  MISSING ARM -- cannot compare")
            continue
        r = compare(label, b2, b3, parent.get(pkey))
        if r:
            out.append(r)

    if out:
        print("\n" + "=" * 78)
        print("SUMMARY")
        print("=" * 78)
        for r in out:
            if r["resolved"]:
                verdict = f"RESOLVED at {r['pct']:+.1f}% -- three nodes faster"
            elif r["excludes_depth_effect"]:
                verdict = (f"still inconclusive, but CI now excludes "
                           f"+{DEPTH_SWEEP_EFFECT}% -- the advantage IS smaller here")
            else:
                verdict = "still inconclusive, and cannot rule out the full effect"
            print(f"  {r['label'].split('--')[0].strip()}: {verdict}")
        print("\n  n=30 was the pre-committed stopping point. Do not re-run these")
        print("  cells again hoping for significance.")

    if args.json and out:
        args.json.write_text(json.dumps(out, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
