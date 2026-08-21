#!/usr/bin/env python3
from __future__ import annotations
import argparse, json, statistics
from collections import defaultdict
from pathlib import Path


def median(vals):
    vals=[v for v in vals if isinstance(v,(int,float))]
    return statistics.median(vals) if vals else None


def fmt(v, digits=2): return "—" if v is None else f"{v:.{digits}f}"

def pct(before, after):
    if before in (None, 0) or after is None: return None
    return (after-before)/before*100

def load(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]

def summarize(rows):
    groups=defaultdict(list)
    for r in rows:
        groups[(r["context_target"],r["concurrency"])].append(r)
    out={}
    for key, rs in groups.items():
        ok=[r for r in rs if r.get("ok")]
        out[key]={
            "requests":len(rs),
            "success_rate":len(ok)/len(rs) if rs else 0,
            "needle_rate":sum(bool(r.get("needle_correct")) for r in ok)/len(ok) if ok else 0,
            "ttft":median([r.get("ttft_s") for r in ok]),
            "e2e":median([r.get("e2e_s") for r in ok]),
            "decode_tps":median([r.get("decode_tps") for r in ok]),
            "aggregate_tps":median([r.get("wave_aggregate_output_tps") for r in ok]),
            "prompt_tokens":median([r.get("prompt_tokens") for r in ok]),
        }
    return out


def latest(results_dir, needle):
    matches=sorted(results_dir.glob(f"*{needle}*/benchmark.jsonl"), key=lambda p:p.stat().st_mtime)
    if not matches: raise SystemExit(f"No result matching *{needle}*/benchmark.jsonl")
    return matches[-1]


def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--results-dir",default="results")
    ap.add_argument("--baseline")
    ap.add_argument("--candidate")
    ap.add_argument("--output")
    a=ap.parse_args()
    rd=Path(a.results_dir)
    bp=Path(a.baseline) if a.baseline else latest(rd,"2spark")
    cp=Path(a.candidate) if a.candidate else latest(rd,"3spark")
    b=summarize(load(bp)); c=summarize(load(cp))
    keys=sorted(set(b)|set(c))
    lines=["# DGX Spark Before / After", "", f"Baseline: `{bp}`  ", f"Candidate: `{cp}`", "",
           "| Context target | Conc. | Needle before | Needle after | TTFT before (s) | TTFT after (s) | TTFT Δ | Decode t/s before | Decode t/s after | Decode Δ | Aggregate t/s before | Aggregate t/s after | Aggregate Δ |",
           "|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|"]
    for key in keys:
        x=b.get(key,{}); y=c.get(key,{})
        ptt=pct(x.get('ttft'),y.get('ttft')); pdec=pct(x.get('decode_tps'),y.get('decode_tps')); pagg=pct(x.get('aggregate_tps'),y.get('aggregate_tps'))
        p=lambda z: "—" if z is None else f"{z:+.1f}%"
        lines.append(f"| {key[0]} | {key[1]} | {x.get('needle_rate',0)*100:.0f}% | {y.get('needle_rate',0)*100:.0f}% | {fmt(x.get('ttft'))} | {fmt(y.get('ttft'))} | {p(ptt)} | {fmt(x.get('decode_tps'))} | {fmt(y.get('decode_tps'))} | {p(pdec)} | {fmt(x.get('aggregate_tps'))} | {fmt(y.get('aggregate_tps'))} | {p(pagg)} |")
    lines += ["", "## Interpretation", "", "- Negative TTFT delta is better.", "- Positive decode and aggregate throughput deltas are better.", "- A speed improvement is not accepted if needle correctness regresses.", "- Compare environment manifests before attributing any delta to node count/parallelism."]
    text="\n".join(lines)+"\n"
    out=Path(a.output) if a.output else rd/"comparison.md"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")

if __name__=="__main__": main()
