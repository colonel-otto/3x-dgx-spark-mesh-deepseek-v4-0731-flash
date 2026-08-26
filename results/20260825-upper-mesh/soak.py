#!/usr/bin/env python3
"""Sustained-load soak against the 3-Spark DSv4 cluster (4-HCA fabric validation).

Drives 8 concurrent streams of varied prompt sizes for a wall-clock budget,
recording per-request latency and every HTTP/connection failure.

Measure-only: no config changes, no restarts.
"""
import json
import os
import random
import statistics
import sys
import threading
import time
import urllib.error
import urllib.request

ENDPOINT = "http://192.168.1.223:8100/v1/chat/completions"
MODEL = "deepseek-v4-flash-0731"
CONCURRENCY = 8
BUDGET_S = float(os.environ.get("SOAK_BUDGET_S", 1200))  # ~20 min
MAX_TOKENS = 256
REQ_TIMEOUT = 300

OUTDIR = os.path.dirname(os.path.abspath(__file__))
RESULTS_PATH = os.path.join(OUTDIR, "soak-results.jsonl")

# Filler corpus -> approx token sizes. ~0.75 tokens/word is the usual English
# ratio, so we size by word count and report the target bucket.
WORDS = (
    "the quick brown fox jumps over a lazy dog while distributed tensors "
    "traverse the fabric between nodes carrying activations and gradients "
    "through remote direct memory access queues in a tightly coupled cluster "
).split()

SIZES = [1000, 4000, 16000]  # target prompt tokens


def make_prompt(target_tokens, rng):
    # ~0.75 words per token for this corpus; overshoot slightly then rely on
    # the model's own tokenizer. Deterministic-ish per request via rng.
    n_words = int(target_tokens * 0.78)
    body = " ".join(rng.choice(WORDS) for _ in range(n_words))
    return (
        "Below is a block of filler text. Read it, then answer the question.\n\n"
        f"{body}\n\n"
        "Question: In two sentences, what is the purpose of a soak test?"
    )


lock = threading.Lock()
records = []
stop_flag = threading.Event()


def log(msg):
    ts = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    print(f"[{ts}] {msg}", flush=True)


def one_request(stream_id, seq, target_tokens, rng):
    payload = {
        "model": MODEL,
        "messages": [{"role": "user", "content": make_prompt(target_tokens, rng)}],
        "max_tokens": MAX_TOKENS,
        "temperature": 0.7,
        "frequency_penalty": 0.3,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(
        ENDPOINT, data=data, headers={"Content-Type": "application/json"}
    )
    t0 = time.time()
    rec = {
        "stream": stream_id,
        "seq": seq,
        "target_tokens": target_tokens,
        "start": t0,
        "start_iso": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t0)),
    }
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT) as resp:
            body = resp.read()
            dt = time.time() - t0
            j = json.loads(body)
            usage = j.get("usage", {})
            rec.update(
                ok=True,
                status=resp.status,
                latency_s=dt,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )
    except urllib.error.HTTPError as e:
        rec.update(
            ok=False,
            status=e.code,
            latency_s=time.time() - t0,
            error=f"HTTPError {e.code}: {e.read()[:400].decode(errors='replace')}",
        )
        log(f"FAILURE stream={stream_id} seq={seq} {rec['error'][:200]}")
    except Exception as e:  # connection reset, timeout, etc.
        rec.update(
            ok=False,
            status=None,
            latency_s=time.time() - t0,
            error=f"{type(e).__name__}: {e}",
        )
        log(f"FAILURE stream={stream_id} seq={seq} {rec['error'][:200]}")

    with lock:
        records.append(rec)
    return rec


def worker(stream_id, deadline):
    rng = random.Random(1000 + stream_id)
    seq = 0
    while time.time() < deadline and not stop_flag.is_set():
        target = SIZES[seq % len(SIZES)]
        one_request(stream_id, seq, target, rng)
        seq += 1
    log(f"stream {stream_id} finished after {seq} requests")


def pct(vals, p):
    if not vals:
        return None
    s = sorted(vals)
    k = (len(s) - 1) * (p / 100.0)
    lo, hi = int(k), min(int(k) + 1, len(s) - 1)
    return s[lo] + (s[hi] - s[lo]) * (k - lo)


def main():
    start = time.time()
    deadline = start + BUDGET_S
    log(f"SOAK START budget={BUDGET_S:.0f}s concurrency={CONCURRENCY} "
        f"endpoint={ENDPOINT} sizes={SIZES} max_tokens={MAX_TOKENS}")

    threads = [
        threading.Thread(target=worker, args=(i, deadline), daemon=True)
        for i in range(CONCURRENCY)
    ]
    for t in threads:
        t.start()

    # progress heartbeat
    while any(t.is_alive() for t in threads):
        time.sleep(30)
        with lock:
            n = len(records)
            nf = sum(1 for r in records if not r["ok"])
        elapsed = time.time() - start
        log(f"progress t={elapsed:6.0f}s requests={n} failures={nf}")

    for t in threads:
        t.join()

    elapsed = time.time() - start
    with lock:
        recs = list(records)

    with open(RESULTS_PATH, "w") as f:
        for r in recs:
            f.write(json.dumps(r) + "\n")

    oks = [r for r in recs if r["ok"]]
    fails = [r for r in recs if not r["ok"]]
    lat = [r["latency_s"] for r in oks]

    summary = {
        "duration_s": round(elapsed, 1),
        "concurrency": CONCURRENCY,
        "total_requests": len(recs),
        "successful": len(oks),
        "failures": len(fails),
        "failure_detail": [f.get("error", "")[:300] for f in fails][:50],
        "latency_s": {
            "p50": pct(lat, 50),
            "p95": pct(lat, 95),
            "p99": pct(lat, 99),
            "min": min(lat) if lat else None,
            "max": max(lat) if lat else None,
            "mean": statistics.mean(lat) if lat else None,
        },
        "by_size": {},
        "total_prompt_tokens": sum(r.get("prompt_tokens") or 0 for r in oks),
        "total_completion_tokens": sum(r.get("completion_tokens") or 0 for r in oks),
    }
    for sz in SIZES:
        sub = [r["latency_s"] for r in oks if r["target_tokens"] == sz]
        ptoks = [r.get("prompt_tokens") for r in oks
                 if r["target_tokens"] == sz and r.get("prompt_tokens")]
        summary["by_size"][str(sz)] = {
            "count": len(sub),
            "p50": pct(sub, 50),
            "p95": pct(sub, 95),
            "p99": pct(sub, 99),
            "actual_prompt_tokens_median": (
                statistics.median(ptoks) if ptoks else None
            ),
        }

    with open(os.path.join(OUTDIR, "soak-summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    log("SOAK COMPLETE")
    print(json.dumps(summary, indent=2), flush=True)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
