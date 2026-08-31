#!/usr/bin/env python3
"""The three remaining engine-A/B cells, on the eugr engine.

Cells (docs/ENGINE-AB-3NODE.md "The cells" table):
  1. decode at 131,072-token context, c=1      (anemll ref: 83.5 tok/s)
  2. prompt-effect pair: code-brief / dense-prose, c=1
                                                (anemll ref: 81.8 / 49.4)
  3. deep concurrency 4 x ~200K                 (anemll ref: 0.9, "unusable")

PROMPT PROVENANCE -- read before comparing anything.

The code-brief prompt is recovered verbatim from the committed harness
`results/20260825-fabric-fix/harness/bench_tp3.py` (its --prompt default, 18
prompt tokens, matching the CSV rows). It is exact.

The dense-prose prompt is NOT recoverable. `ours-bench.py` was never committed
and no doc records its text -- only that the prompt was ~51 tokens and produced
49.4 tok/s. The prompt below is a RECONSTRUCTION built to the recorded shape
(~51 tokens of continuous natural-language prose, no code, no lists). It is
therefore NOT byte-comparable to the 2026-08-21 anemll row.

What that costs, precisely: the anemll-vs-eugr *dense-prose* number is not a
matched comparison and must not be quoted as one. What survives is the
within-engine ratio -- code-brief vs dense-prose measured here, on one engine,
minutes apart, with both prompts recorded in this file. That ratio is the thing
the cell exists to test (MTP/DSpark acceptance depends on prompt type), and it
is self-contained.

Sampling matches bench-miaai so the numbers sit in the same family: streamed,
temperature 0.6 / top_p 0.95, min=max tokens, ignore_eos, thinking off, unique
nonce per long request to defeat the prefix cache.
"""
import argparse
import json
import statistics
import threading
import time
import urllib.request

BASE_DEFAULT = "http://127.0.0.1:8100/v1"
MODEL_DEFAULT = "deepseek-v4-flash-eugr-ab"

# Verbatim from results/20260825-fabric-fix/harness/bench_tp3.py --prompt default.
CODE_BRIEF = ("Write a Python function that merges two sorted "
              "lists. Explain briefly.")

# RECOVERED VERBATIM from repo git history (commit b078eb4, the benchmark table
# row that produced the 49.4 tok/s anemll number). The README elided it with an
# ellipsis, which is why it was believed lost. Byte-exact to the 2026-08-21 row.
DENSE_PROSE = ("Write a detailed technical explanation of how pipeline parallelism "
               "differs from tensor parallelism in large language model inference.")


def post(url, body, timeout=3600):
    req = urllib.request.Request(
        url, data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.load(r)


def count_tokens(base, model, text):
    url = base.removesuffix("/v1") + "/tokenize"
    return post(url, {"model": model, "prompt": text})["count"]


def build_long_prompt(base, model, target, nonce):
    """Grow a unique filler prompt to >= target tokens (prefix-cache hostile)."""
    unit = "benchmark context datum "
    text = "unique request " + nonce + " " + unit * max(1, target // 3)
    while True:
        n = count_tokens(base, model, text)
        if n >= target:
            return text, n
        text += unit * max(1, (target - n) // 3)


def stream_one(base, model, prompt, max_tokens, out, idx, timeout=3600, sampling=None):
    body = {
        "model": model,
        "messages": [{"role": "user", "content": prompt}],
        "stream": True, "stream_options": {"include_usage": True},
        "temperature": 0.6, "top_p": 0.95,
        "max_tokens": max_tokens, "min_tokens": max_tokens, "ignore_eos": True,
        "chat_template_kwargs": {"thinking": False},
    }
    if sampling:  # ours-bench.py conditions for the prompt-effect cell
        body.update(sampling)
        for k in sampling.get("_drop", []):
            body.pop(k, None)
        body.pop("_drop", None)
    req = urllib.request.Request(
        base + "/chat/completions", data=json.dumps(body).encode(),
        headers={"Content-Type": "application/json"})
    started = time.perf_counter()
    first = None
    usage = None
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            for raw in resp:
                line = raw.decode().strip()
                if not line.startswith("data: ") or line == "data: [DONE]":
                    continue
                ev = json.loads(line[6:])
                ch = ev.get("choices") or []
                d = ch[0].get("delta", {}) if ch else {}
                if first is None and (d.get("content") or d.get("reasoning")
                                      or d.get("reasoning_content")):
                    first = time.perf_counter()
                if ev.get("usage"):
                    usage = ev["usage"]
    except Exception as exc:  # noqa: BLE001
        out[idx] = {"error": repr(exc)}
        return
    done = time.perf_counter()
    tok = (usage or {}).get("completion_tokens", 0)
    out[idx] = {
        "ttft_s": (first or done) - started,
        "decode_tok_s": tok / max(1e-3, done - (first or done)),
        "output_tokens": tok,
        "prompt_tokens": (usage or {}).get("prompt_tokens", 0),
    }


def single(base, model, prompt, max_tokens, reps, label, sampling=None):
    rows = []
    for i in range(reps):
        out = {}
        stream_one(base, model, prompt, max_tokens, out, 0, sampling=sampling)
        r = out[0]
        if "error" in r:
            print("  " + label + " rep " + str(i) + ": ERROR " + r["error"], flush=True)
            continue
        rows.append(r)
        print("  {} rep {}: decode={:.1f} tok/s ttft={:.0f}ms prompt_tok={} out={}".format(
            label, i, r["decode_tok_s"], r["ttft_s"] * 1000,
            r["prompt_tokens"], r["output_tokens"]), flush=True)
    if not rows:
        return None
    return {
        "label": label,
        "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in rows), 1),
        "ttft_ms": round(statistics.median(r["ttft_s"] for r in rows) * 1000),
        "prompt_tokens": rows[0]["prompt_tokens"],
        "reps": len(rows),
        "min": round(min(r["decode_tok_s"] for r in rows), 1),
        "max": round(max(r["decode_tok_s"] for r in rows), 1),
    }


def concurrent(base, model, prompts, max_tokens, label):
    out = {}
    threads = [threading.Thread(target=stream_one,
                                args=(base, model, p, max_tokens, out, i))
               for i, p in enumerate(prompts)]
    t0 = time.perf_counter()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.perf_counter() - t0
    ok = [r for r in out.values() if "error" not in r]
    errs = [r for r in out.values() if "error" in r]
    if not ok:
        print("  " + label + ": ALL " + str(len(errs)) + " REQUESTS FAILED", flush=True)
        return {"label": label, "failed": len(errs),
                "errors_sample": [r["error"] for r in errs[:2]]}
    total = sum(r["output_tokens"] for r in ok)
    res = {
        "label": label,
        "streams": len(prompts),
        "ok": len(ok), "errors": len(errs),
        "wall_s": round(wall, 1),
        "aggregate_tok_s": round(total / max(1e-3, wall), 2),
        "median_decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in ok), 2),
        "median_ttft_s": round(statistics.median(r["ttft_s"] for r in ok), 1),
        "prompt_tokens": ok[0]["prompt_tokens"],
    }
    print("  {}: agg={} tok/s median_decode={} ttft={}s wall={}s prompt_tok={} errors={}".format(
        label, res["aggregate_tok_s"], res["median_decode_tok_s"],
        res["median_ttft_s"], res["wall_s"], res["prompt_tokens"],
        res["errors"]), flush=True)
    return res


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-url", default=BASE_DEFAULT)
    ap.add_argument("--model", default=MODEL_DEFAULT)
    ap.add_argument("--out", default="remaining-cells.json")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--cells", default="prompt,131k,deep",
                    help="comma-separated subset: prompt, 131k, deep")
    args = ap.parse_args()
    base, model = args.base_url, args.model
    want = set(c.strip() for c in args.cells.split(","))
    results = {}

    if "prompt" in want:
        print("== cell: prompt-effect pair (c=1, 256 out) ==", flush=True)
        OURS_BENCH = {"temperature": 0, "_drop": ["top_p", "min_tokens", "ignore_eos"]}
        results["code_brief"] = single(base, model, CODE_BRIEF, 256,
                                       args.reps, "code-brief", sampling=OURS_BENCH)
        results["dense_prose"] = single(base, model, DENSE_PROSE, 256,
                                        args.reps, "dense-prose", sampling=OURS_BENCH)
        cb, dp = results.get("code_brief"), results.get("dense_prose")
        if cb and dp and dp["decode_tok_s"]:
            ratio = cb["decode_tok_s"] / dp["decode_tok_s"]
            results["prompt_effect_ratio"] = round(ratio, 3)
            print("  prompt effect: code-brief / dense-prose = {:.2f}x "
                  "(anemll recorded 1.65x)".format(ratio), flush=True)

    if "131k" in want:
        # Every rep gets a UNIQUE 131K prompt. Reusing one prompt lets the
        # prefix cache serve reps 2..N: observed TTFT 58,742ms on the cold rep
        # then 1,262ms on the next, i.e. the later reps measure a cache hit and
        # not a 131K prefill. The anemll reference (83.5) is a COLD number, so
        # a warmed median would not be comparable to it.
        print("== cell: decode at 131,072-token context (c=1, 128 out) ==", flush=True)
        reps131 = max(3, args.reps // 2)
        rows = []
        for i in range(reps131):
            p, n = build_long_prompt(
                base, model, 131072,
                "ctx131k-" + str(int(time.time())) + "-r" + str(i))
            out = {}
            stream_one(base, model, p, 128, out, 0)
            r = out[0]
            if "error" in r:
                print("  131k rep " + str(i) + ": ERROR " + r["error"], flush=True)
                continue
            rows.append(r)
            print("  131k-decode rep {} (cold, {} tok): decode={:.1f} tok/s ttft={:.0f}ms".format(
                i, n, r["decode_tok_s"], r["ttft_s"] * 1000), flush=True)
        if rows:
            results["ctx_131k"] = {
                "label": "131k-decode-cold",
                "decode_tok_s": round(statistics.median(r["decode_tok_s"] for r in rows), 1),
                "ttft_ms": round(statistics.median(r["ttft_s"] for r in rows) * 1000),
                "prompt_tokens": rows[0]["prompt_tokens"],
                "reps": len(rows),
                "min": round(min(r["decode_tok_s"] for r in rows), 1),
                "max": round(max(r["decode_tok_s"] for r in rows), 1),
                "note": "unique prompt per rep; every rep is a cold 131K prefill",
            }

    if "deep" in want:
        print("== cell: deep concurrency 4 x ~200K (128 out each) ==", flush=True)
        prompts = []
        for i in range(4):
            p, n = build_long_prompt(
                base, model, 200000,
                "deep200k-" + str(int(time.time())) + "-s" + str(i))
            prompts.append(p)
            print("  stream " + str(i) + ": " + str(n) + " prompt tokens", flush=True)
        results["deep_4x200k"] = concurrent(base, model, prompts, 128, "4x200k")

    results["_meta"] = {
        "base_url": base,
        "model": model,
        "code_brief_prompt": CODE_BRIEF,
        "dense_prose_prompt": DENSE_PROSE,
        "dense_prose_is_reconstruction": True,
        "dense_prose_note": (
            "ours-bench.py was never committed and no doc records its prompt "
            "text; this is a reconstruction to the recorded ~51-token prose "
            "shape. The cross-engine dense-prose number is NOT matched; only "
            "the within-engine code/prose ratio is."),
    }
    with open(args.out, "w") as f:
        json.dump(results, f, indent=2)
    print("\nwrote " + args.out, flush=True)


if __name__ == "__main__":
    main()
