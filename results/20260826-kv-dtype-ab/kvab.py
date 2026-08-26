#!/usr/bin/env python3
"""KV-cache dtype A/B harness: nvfp4_ds_mla vs fp8_ds_mla.

Issue #16. Both dtypes use the same 584-byte sparse-MLA envelope on
DeepSeek-V4, so memory is identical and the choice is free. This measures
whether QUALITY or SPEED differ.

Subcommands, all deterministic across arms (same prompts, same seeds, same
trial counts, so the only variable is the dtype):

  gate     correctness gate -- a broken config serves fluent nonsense, not an error
  warm     exercise the shapes so a JIT compile does not land in a measurement
  quality  needle retrieval at 4K/32K/128K/256K x 3 depths x N trials
  speed    decode tok/s at concurrency 1/4/8, median of N runs, plus TTFT

Usage: kvab.py <arm-label> {gate|warm|quality|speed} [--trials N] [--runs N]
"""
import argparse, concurrent.futures as cf, json, re, statistics, sys, time, urllib.request

BASE = "http://192.168.1.223:8100"
MODEL = "deepseek-v4-flash-0731"

# ---------------------------------------------------------------- filler text
# Varied technical prose. Deliberately NOT repeated text: repeated filler is
# highly compressible for the sparse-attention indexer and would make the
# retrieval task easier than real use.
FILLER = [
    "The scheduler batches prefill and decode work into a single engine step.",
    "Each attention layer writes its keys and values into the paged cache.",
    "Block tables map logical positions onto physical cache blocks.",
    "Speculative decoding proposes several tokens before the target verifies them.",
    "Tensor parallelism shards attention heads across the participating devices.",
    "The router selects a small subset of experts for every token it processes.",
    "Quantized weights are dequantized inside the kernel just before the matmul.",
    "Prefix caching lets a shared conversation prefix skip recomputation entirely.",
    "Continuous batching admits new requests without draining the current batch.",
    "The sampler applies penalties and temperature after the logits are produced.",
    "A paged allocator hands out fixed-size blocks and tracks their reference counts.",
    "Chunked prefill splits one long prompt across several scheduler iterations.",
    "The indexer scores candidate blocks before the sparse kernel reads any of them.",
    "Rotary embeddings are applied to the query and key projections in place.",
    "Weight loading streams shards from disk and validates every tensor checksum.",
    "The draft model proposes tokens that the target model accepts or rejects.",
    "Grouped query attention lets several query heads share one key-value head.",
    "An all-reduce collects partial sums from every rank before the residual add.",
    "The tokenizer normalizes whitespace before it emits any subword identifier.",
    "Guided decoding masks logits so only grammar-valid tokens remain reachable.",
]

DEPTH_FRACS = [("early", 0.10), ("mid", 0.50), ("late", 0.90)]

# Distinct code per (length, depth, trial) so no answer can leak between cells
# and a lucky guess cannot repeat. Deterministic, so both arms see identical
# prompts.
WORDS_A = ["MAGENTA", "OBSIDIAN", "CRIMSON", "VERDANT", "AZURE", "AMBER",
           "SLATE", "COBALT", "SIENNA", "IVORY", "TEAL", "RUSSET"]
WORDS_B = ["HERON", "LANTERN", "FULCRUM", "MERIDIAN", "QUARRY", "TRELLIS",
           "CANYON", "BASTION", "PLINTH", "HARROW", "VESSEL", "KEYSTONE"]

CJK = re.compile(r"[　-鿿가-힯]")
SPECIAL = re.compile(r"<\|.*?\|>|�|<pad>|<s>|</s>")


def codes_for(length, trial):
    """Deterministic, distinct triple for this cell. Identical across arms."""
    out = []
    for i, (name, _) in enumerate(DEPTH_FRACS):
        h = (length * 7919 + trial * 104729 + i * 1299709)
        a = WORDS_A[h % len(WORDS_A)]
        b = WORDS_B[(h // 12) % len(WORDS_B)]
        n = 1000 + (h // 144) % 9000
        out.append((name, "%s-%s-%d" % (a, b, n)))
    return out


def build_needle(length, trial, salt=None):
    """salt: when set, prepended so the prompt is a PREFIX-CACHE MISS.

    Without it, re-running the same (length, trial) against an engine that has
    already served it returns from prefix cache in ~1s instead of ~135s, which
    does not re-exercise the KV path at all and would silently fake a pass.
    The salt is identical across arms for a given (length, trial, salt), so the
    A/B stays matched.
    """
    target_words = int(length / 1.3)
    lines, n = [], 0
    while n < target_words:
        lines.append(FILLER[len(lines) % len(FILLER)])
        n += len(lines[-1].split())
    triple = codes_for(length, trial)
    for (name, val), (_, frac) in zip(triple, DEPTH_FRACS):
        idx = min(int(len(lines) * frac), len(lines) - 1)
        lines.insert(idx, "Note carefully: the access code at this point is %s." % val)
    q = ("\n\nQuestion: three access codes appear above. List all three, exactly as "
         "written, one per line, and nothing else.")
    body = "\n".join(lines)
    if salt is not None:
        # Leading salt -> the shared prefix diverges at token ~0, so the whole
        # prompt must be re-prefilled.
        body = ("Session reference %s. Ignore this line; it carries no access code.\n"
                % salt) + body
    return body + q, triple


def filler_prompt(approx_tokens):
    """Plain filler of ~N tokens, no needles. For warmup and speed runs."""
    target_words = int(approx_tokens / 1.3)
    lines, n = [], 0
    while n < target_words:
        lines.append(FILLER[len(lines) % len(FILLER)])
        n += len(lines[-1].split())
    return "\n".join(lines)


def post(prompt, max_tokens, temperature=0.0, stream=False, timeout=600):
    body = {"model": MODEL, "messages": [{"role": "user", "content": prompt}],
            "max_tokens": max_tokens, "temperature": temperature, "seed": 1234}
    if stream:
        body["stream"] = True
        body["stream_options"] = {"include_usage": True}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time()
    if not stream:
        d = json.loads(urllib.request.urlopen(req, timeout=timeout).read())
        return {"text": d["choices"][0]["message"]["content"] or "",
                "prompt_tokens": d["usage"]["prompt_tokens"],
                "completion_tokens": d["usage"]["completion_tokens"],
                "elapsed": time.time() - t0, "ttft": None}
    ttft = None
    text = []
    ptok = ctok = 0
    with urllib.request.urlopen(req, timeout=timeout) as r:
        for raw in r:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data: "):
                continue
            payload = line[6:]
            if payload == "[DONE]":
                break
            try:
                d = json.loads(payload)
            except Exception:
                continue
            if d.get("usage"):
                ptok = d["usage"].get("prompt_tokens", ptok)
                ctok = d["usage"].get("completion_tokens", ctok)
            for ch in d.get("choices", []):
                piece = (ch.get("delta") or {}).get("content")
                if piece:
                    if ttft is None:
                        ttft = time.time() - t0
                    text.append(piece)
    el = time.time() - t0
    joined = "".join(text)
    if not ctok:
        ctok = max(1, len(joined) // 4)
    out = {"text": joined, "prompt_tokens": ptok, "completion_tokens": ctok,
           "elapsed": el, "ttft": ttft, "decode_tps": None}
    if ttft is not None and el > ttft and ctok > 1:
        out["decode_tps"] = (ctok - 1) / (el - ttft)
    return out


def garble(text):
    flags = []
    if CJK.search(text):
        flags.append("CJK")
    if SPECIAL.search(text):
        flags.append("SPECIAL")
    words = text.split()
    if len(words) > 12:
        for w in set(words):
            if len(w) > 3 and words.count(w) > len(words) * 0.35:
                flags.append("REPEAT:" + w[:16])
                break
    return flags


# ------------------------------------------------------------------- commands
def cmd_gate(arm, args):
    ok_all = True
    for q, expect in [("What is 17 times 23?", "391"),
                      ("What is the capital of France? Answer with one word.", "Paris")]:
        r = post(q, 400)
        ok = expect.lower() in r["text"].lower()
        ok_all = ok_all and ok
        print("[%s] GATE %r -> %s | reply=%r" %
              (arm, q, "PASS" if ok else "FAIL", r["text"][:200]), flush=True)
    return 0 if ok_all else 1


def cmd_warm(arm, args):
    """Exercise every prompt length and concurrency we will later measure."""
    for n in [512, 2048, 4096, 8192, 16384, 32768, 131072, 262144]:
        p = filler_prompt(n)
        for rep in range(3):
            r = post(p + "\n\nReply with the single word: ok.", 32)
            print("[%s] warm len~%d rep%d: %.2fs ptok=%d" %
                  (arm, n, rep, r["elapsed"], r["prompt_tokens"]), flush=True)
    # decode-shaped warm: same prompt and max_tokens the speed run uses
    for conc in [1, 4, 8]:
        for rep in range(2):
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [ex.submit(post, DECODE_PROMPT + " (variant %d)" % i, 512,
                                  0.0, True) for i in range(conc)]
                rs = [f.result() for f in futs]
            tps = [x["decode_tps"] for x in rs if x["decode_tps"]]
            print("[%s] warm conc=%d rep%d: per-stream %.2f tok/s" %
                  (arm, conc, rep, statistics.median(tps) if tps else -1), flush=True)
    return 0


def cmd_quality(arm, args):
    for length in args.lengths:
        for trial in range(args.trials):
            # Salt varies per trial as well, so trials inside one run cannot
            # prefix-cache off each other either.
            cell_salt = None if args.salt is None else "%s-%04d" % (args.salt, trial)
            prompt, triple = build_needle(length, trial, salt=cell_salt)
            try:
                r = post(prompt, 200)
            except Exception as e:
                print("[%s] len=%d trial=%d REQUEST FAILED: %s" %
                      (arm, length, trial, str(e)[:160]), flush=True)
                with open(args.out, "a") as f:
                    f.write(json.dumps({"arm": arm, "length": length, "trial": trial,
                                        "error": str(e)[:200]}) + "\n")
                continue
            reply = r["text"]
            per_depth = dict((name, (val in reply)) for name, val in triple)
            g = garble(reply)
            npass = sum(1 for v in per_depth.values() if v)
            status = "PASS" if npass == 3 and not g else "FAIL"
            row = {"arm": arm, "length": length, "trial": trial,
                   "prompt_tokens": r["prompt_tokens"], "elapsed": r["elapsed"],
                   "depths": per_depth, "garble": g, "status": status,
                   "codes": dict(triple), "reply": reply[:500]}
            print("[%s] len~%7d t%d | %7d ptok | %7.1fs | early=%s mid=%s late=%s | "
                  "garble %s | %s" %
                  (arm, length, trial, r["prompt_tokens"], r["elapsed"],
                   "Y" if per_depth["early"] else "N",
                   "Y" if per_depth["mid"] else "N",
                   "Y" if per_depth["late"] else "N",
                   ",".join(g) if g else "none", status), flush=True)
            if status == "FAIL":
                missing = [n + "=" + v for (n, v) in triple if not per_depth[n]]
                print("    missing: %s" % missing, flush=True)
                print("    reply[:300]: %r" % reply[:300], flush=True)
            with open(args.out, "a") as f:
                f.write(json.dumps(row) + "\n")
    return 0


DECODE_PROMPT = ("Write a detailed technical explanation of how paged attention "
                 "manages key-value cache memory in a large language model serving "
                 "system. Cover block allocation, block tables, reference counting, "
                 "copy-on-write for shared prefixes, fragmentation, and eviction. "
                 "Be thorough and write continuous prose.")


def cmd_speed(arm, args):
    results = {}
    for conc in [1, 4, 8]:
        per_run = []
        for run in range(args.runs):
            with cf.ThreadPoolExecutor(max_workers=conc) as ex:
                futs = [ex.submit(post, DECODE_PROMPT + " (variant %d)" % i, 512,
                                  0.0, True) for i in range(conc)]
                rs = [f.result() for f in futs]
            tps = [x["decode_tps"] for x in rs if x["decode_tps"]]
            ttfts = [x["ttft"] for x in rs if x["ttft"]]
            per_stream = statistics.median(tps) if tps else None
            agg = sum(tps) if tps else None
            ttft = statistics.median(ttfts) if ttfts else None
            per_run.append({"run": run, "per_stream_tps": per_stream,
                            "aggregate_tps": agg, "ttft": ttft,
                            "ctok": [x["completion_tokens"] for x in rs]})
            print("[%s] conc=%d run%d: per-stream %.2f tok/s | agg %.2f | ttft %.0f ms" %
                  (arm, conc, run, per_stream, agg, ttft * 1000), flush=True)
        med_ps = statistics.median([p["per_stream_tps"] for p in per_run])
        med_agg = statistics.median([p["aggregate_tps"] for p in per_run])
        med_ttft = statistics.median([p["ttft"] for p in per_run])
        results[conc] = {"runs": per_run, "median_per_stream_tps": med_ps,
                         "median_aggregate_tps": med_agg, "median_ttft_s": med_ttft}
        print("[%s] conc=%d MEDIAN per-stream %.2f tok/s | agg %.2f tok/s | ttft %.0f ms" %
              (arm, conc, med_ps, med_agg, med_ttft * 1000), flush=True)
    with open(args.out, "w") as f:
        json.dump({"arm": arm,
                   "results": dict((str(k), v) for k, v in results.items())}, f, indent=2)
    return 0


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("arm")
    ap.add_argument("cmd", choices=["warm", "quality", "speed", "gate"])
    ap.add_argument("--trials", type=int, default=3)
    ap.add_argument("--runs", type=int, default=5)
    ap.add_argument("--lengths", type=int, nargs="*",
                    default=[4000, 32000, 128000, 256000])
    ap.add_argument("--out", default=None)
    ap.add_argument("--salt", default=None,
                    help="prefix-cache buster; identical across arms so the A/B stays matched")
    a = ap.parse_args()
    if a.out is None:
        a.out = "%s-%s.%s" % (a.arm, a.cmd, "jsonl" if a.cmd == "quality" else "json")
    sys.exit({"warm": cmd_warm, "quality": cmd_quality,
              "speed": cmd_speed, "gate": cmd_gate}[a.cmd](a.arm, a))
