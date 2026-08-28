#!/usr/bin/env python3
"""Numerical parity gate: prompt-logprob perplexity against a reference arm.

Answers a question no existing harness in this repo answers: **is the served
model NUMERICALLY correct**, as opposed to merely alive and fluent?

Every quality test we have (RULER-lite, the tool battery, the garble sweep) is
pass/fail BEHAVIOURAL. They catch a model that is broken. They do not catch a
model that is subtly wrong, because a partially-corrupted transformer still
writes grammatical English. That is not hypothetical here: stock TP=3 computes
`8 // 3 == 2` and silently drops 6 of 8 attention groups, and the result is
FLUENT NONSENSE that passes casual inspection. See docs/patch.md.

Perplexity over teacher-forced prompt logprobs is the cheapest test that fails
loudly in that situation. It reads the model's own probability for every token
of a FIXED text, so it is sensitive to any change in the logits -- including
changes far too small to flip an argmax and therefore invisible to a
text-diff comparison.

## What this measures, precisely

`prompt_logprobs` makes the server score a prompt it is given rather than one
it generates. No sampling is involved, so there is no temperature, no seed, and
no MTP: the number is a deterministic function of the weights, the shard
layout, and the kernels. Two configurations that are numerically equivalent
return the same perplexity to several decimal places. A dropped attention group
moves it enormously.

## How to use it

Two arms, one variable (the thing you are validating), same corpus:

    # reference arm -- a configuration you trust
    python3 logprob_parity.py --url http://HOST:8100/v1 --out ref.json

    # candidate arm -- after a rebuild, a patch, a TP change
    python3 logprob_parity.py --url http://HOST:8100/v1 --out cand.json

    # verdict
    python3 logprob_parity.py --compare ref.json cand.json

TP=1 is the strongest reference because it does no sharding at all, so any
sharding bug is absent by construction. It does not fit on one Spark for this
checkpoint, so in practice the useful comparisons are: before-vs-after an image
rebuild, TP=2-vs-TP=3, and patched-vs-unpatched.

## Traps this harness is built to avoid

1. CHAT ENDPOINT. `prompt_logprobs` is a completions-API parameter. vLLM
   rejected it for chat (vllm-project/vllm#5264), and a chat template would
   change the token sequence anyway. This uses /v1/completions only.

2. PREFIX CACHE. A cache hit returns stored logprobs rather than recomputing
   them, which would compare an arm against the OTHER arm's cached numbers and
   report perfect parity for a broken build. Every passage gets a unique salt
   header, and `cached_tokens` is asserted to be 0 on every request.

3. max_tokens=0 IS NOT ALWAYS ACCEPTED. Some builds reject a zero-length
   generation. This asks for 1 token and ignores it; the score comes from
   `prompt_logprobs`, not from what is generated.

4. FIXED CORPUS, NOT RANDOM. Perplexity is only comparable across arms if the
   text is identical. The corpus is embedded in this file rather than fetched,
   so a network failure cannot silently change what is being scored.

5. DOMAIN COVERAGE. A degenerate MoE routing bug can collapse one domain while
   others hold, so the corpus spans prose, code, math, structured data and
   non-English rather than being one kind of text. Per-passage results are
   reported so a single-domain collapse is visible instead of averaged away.
"""

from __future__ import annotations

import argparse
import json
import math
import statistics
import sys
import urllib.error
import urllib.request
from datetime import datetime, timezone

# ---------------------------------------------------------------------------
# Fixed corpus. Embedded deliberately (trap 4): the bytes scored must not vary
# between arms, and a download failure must not be able to change them.
#
# Five domains (trap 5). A dead expert shard or degenerate routing typically
# hurts one domain far more than the others, which an aggregate would hide.
# ---------------------------------------------------------------------------
CORPUS: dict[str, str] = {
    "prose": (
        "The question of whether a distributed system is correct rarely has a "
        "single answer. A cluster can pass every liveness check, respond to "
        "every probe, and still return results that are quietly wrong. What "
        "makes such failures difficult is not their severity but their "
        "plausibility: the output looks exactly like the output of a working "
        "system, and only a comparison against a known reference reveals the "
        "difference. This is why operators who have been burned once tend to "
        "build numerical gates rather than behavioural ones."
    ),
    "code": (
        "def partition_experts(global_num_experts: int, ep_size: int, ep_rank: int) -> int:\n"
        "    \"\"\"Return the number of experts owned by this rank.\n"
        "\n"
        "    Experts are distributed evenly; any remainder goes to the lower ranks,\n"
        "    so 256 experts across 3 ranks yields 86, 85, 85.\n"
        "    \"\"\"\n"
        "    base = global_num_experts // ep_size\n"
        "    remainder = global_num_experts % ep_size\n"
        "    return base + 1 if ep_rank < remainder else base\n"
        "\n"
        "\n"
        "assert partition_experts(256, 3, 0) == 86\n"
        "assert partition_experts(256, 3, 1) == 85\n"
        "assert sum(partition_experts(256, 3, r) for r in range(3)) == 256\n"
    ),
    "math": (
        "Let A be an n-by-n matrix partitioned across p ranks along its column "
        "dimension, so that each rank holds n/p columns. A ring all-reduce over "
        "p ranks completes in 2(p-1) communication steps, each transferring "
        "1/p of the buffer. The total bytes moved per rank is therefore "
        "2(p-1)/p times the buffer size, which approaches 2B as p grows, while "
        "the number of sequential steps grows without bound. For p = 2 the cost "
        "is 2 steps and B bytes; for p = 3 it is 4 steps and 4B/3 bytes. "
        "Latency-bound workloads feel the step count, not the byte count."
    ),
    "structured": (
        '{"cluster": {"nodes": 3, "parallelism": {"tensor": 3, "pipeline": 1, '
        '"expert": false}, "fabric": {"type": "roce", "topology": "point-to-point", '
        '"switch": false, "gdr": false}, "engine": {"max_model_len": 1048576, '
        '"max_num_seqs": 32, "kv_cache_dtype": "nvfp4_ds_mla", '
        '"speculative": {"method": "dspark", "num_speculative_tokens": 5}}, '
        '"checks": ["ruler", "tools", "garble", "logprob_parity"]}}'
    ),
    "multilingual": (
        "Un système distribué peut sembler correct alors qu'il ne l'est pas. "
        "分散システムは、正しく動作しているように見えても、実際には誤った結果を返すことがあります。 "
        "Ein verteiltes System kann fehlerhafte Ergebnisse liefern, ohne dass "
        "ein einziger Fehler protokolliert wird. La única forma de detectarlo "
        "es comparar contra una referencia conocida."
    ),
}


def score_passage(url: str, model: str, name: str, text: str, salt: str,
                  timeout: int) -> dict:
    """Teacher-force one passage and return its mean NLL and perplexity.

    The salt header (trap 2) makes the prompt unique per run so the prefix
    cache cannot serve a stored result. It is scored along with the passage;
    since it is identical across arms, it cancels in any comparison.
    """
    prompt = f"[parity {salt}] {text}"
    body = json.dumps({
        "model": model,
        "prompt": prompt,
        # Trap 3: ask for one token rather than zero. It is discarded; the
        # measurement is entirely in prompt_logprobs.
        "max_tokens": 1,
        "temperature": 0,
        # 1 is enough: we only read the logprob of the token actually present.
        "prompt_logprobs": 1,
        "stream": False,
    }).encode()
    request = urllib.request.Request(
        url.rstrip("/") + "/completions",
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.load(response)

    choice = payload["choices"][0]
    prompt_logprobs = choice.get("prompt_logprobs")
    if not prompt_logprobs:
        raise RuntimeError(
            f"{name}: server returned no prompt_logprobs. This build may not "
            "support the parameter on /v1/completions."
        )

    usage = payload.get("usage") or {}
    cached = (usage.get("prompt_tokens_details") or {}).get("cached_tokens", 0) or 0
    # Trap 2: a cache hit means these numbers were not recomputed by this arm.
    if cached:
        raise RuntimeError(
            f"{name}: {cached} cached prompt tokens. The salt failed to bust "
            "the prefix cache, so this measurement reflects stored logprobs "
            "and cannot be compared across arms."
        )

    # The first entry is null: the initial token has no preceding context and
    # therefore no conditional probability.
    logprobs: list[float] = []
    for entry in prompt_logprobs:
        if entry is None:
            continue
        # Each entry maps token id -> {"logprob": float, ...}. Exactly one of
        # them is the token actually in the prompt; vLLM marks it with rank 1.
        chosen = None
        for info in entry.values():
            if info.get("rank") == 1:
                chosen = info["logprob"]
                break
        if chosen is None:
            # Fall back to the maximum, which is the rank-1 entry by definition
            # when rank is not reported by this build.
            chosen = max(info["logprob"] for info in entry.values())
        logprobs.append(chosen)

    if not logprobs:
        raise RuntimeError(f"{name}: no scored tokens")

    mean_nll = -sum(logprobs) / len(logprobs)
    return {
        "passage": name,
        "scored_tokens": len(logprobs),
        "prompt_tokens": usage.get("prompt_tokens"),
        "mean_nll": mean_nll,
        "perplexity": math.exp(mean_nll),
        "sum_logprob": sum(logprobs),
    }


def run(url: str, model: str, reps: int, timeout: int) -> dict:
    results: list[dict] = []
    for name, text in CORPUS.items():
        per_rep = []
        for rep in range(reps):
            salt = f"{name}-{rep}"
            per_rep.append(score_passage(url, model, name, text, salt, timeout))

        # Determinism check. prompt_logprobs involves no sampling, so repeated
        # scoring of the same text must return the same number. If it does not,
        # something is nondeterministic in the serving path and every parity
        # number below is untrustworthy -- report it rather than averaging it.
        nlls = [r["mean_nll"] for r in per_rep]
        spread = max(nlls) - min(nlls) if len(nlls) > 1 else 0.0
        stdev = statistics.stdev(nlls) if len(nlls) > 1 else 0.0
        record = dict(per_rep[0])
        record["reps"] = reps
        record["all_nlls"] = nlls
        record["all_perplexities"] = [math.exp(n) for n in nlls]
        record["mean_nll"] = statistics.median(nlls)
        record["mean_nll_avg"] = statistics.mean(nlls)
        record["nll_stdev"] = stdev
        record["perplexity"] = math.exp(record["mean_nll"])
        record["nll_spread_across_reps"] = spread
        record["nll_relative_spread_pct"] = (spread / record["mean_nll"]) * 100.0 if record["mean_nll"] != 0 else 0.0
        record["deterministic"] = spread < 1e-6
        results.append(record)
        status = "OK " if record["deterministic"] else "NONDET"
        print(
            f"  {status} {name:<14} ppl={record['perplexity']:9.4f} "
            f"nll={record['mean_nll']:.6f} spread={spread:.6f} ({record['nll_relative_spread_pct']:.2f}%) tokens={record['scored_tokens']}",
            flush=True,
        )

    total_logprob = sum(r["sum_logprob"] for r in results)
    total_tokens = sum(r["scored_tokens"] for r in results)
    aggregate_nll = -total_logprob / total_tokens
    return {
        "captured_utc": datetime.now(timezone.utc).isoformat(),
        "url": url,
        "model": model,
        "reps": reps,
        "aggregate": {
            "mean_nll": aggregate_nll,
            "perplexity": math.exp(aggregate_nll),
            "scored_tokens": total_tokens,
        },
        "passages": results,
    }


def compare(ref_path: str, cand_path: str, tolerance: float) -> int:
    with open(ref_path, encoding="utf-8") as handle:
        ref = json.load(handle)
    with open(cand_path, encoding="utf-8") as handle:
        cand = json.load(handle)

    ref_by_name = {p["passage"]: p for p in ref["passages"]}
    cand_by_name = {p["passage"]: p for p in cand["passages"]}

    print(f"reference : {ref_path}  ({ref['captured_utc']})")
    print(f"candidate : {cand_path}  ({cand['captured_utc']})")
    print()
    print(f"{'passage':<16}{'ref ppl':>12}{'cand ppl':>12}{'delta %':>10}  verdict")

    failures: list[str] = []
    for name in ref_by_name:
        if name not in cand_by_name:
            failures.append(f"{name}: missing from candidate")
            continue
        r = ref_by_name[name]
        c = cand_by_name[name]
        delta = (c["perplexity"] - r["perplexity"]) / r["perplexity"] * 100.0
        ok = abs(delta) <= tolerance
        if not ok:
            failures.append(
                f"{name}: perplexity moved {delta:+.3f}% (tolerance {tolerance}%)"
            )
        if not c.get("deterministic", True):
            failures.append(f"{name}: candidate scoring was NONDETERMINISTIC")
        print(
            f"{name:<16}{r['perplexity']:>12.4f}{c['perplexity']:>12.4f}"
            f"{delta:>+9.3f}%  {'PASS' if ok else 'FAIL'}"
        )

    ra = ref["aggregate"]["perplexity"]
    ca = cand["aggregate"]["perplexity"]
    agg_delta = (ca - ra) / ra * 100.0
    print(f"{'AGGREGATE':<16}{ra:>12.4f}{ca:>12.4f}{agg_delta:>+9.3f}%")
    print()

    if failures:
        print("VERDICT: FAIL")
        for line in failures:
            print(f"  - {line}")
        print()
        print(
            "A perplexity shift means the two arms are not numerically "
            "equivalent. Text-level output can still look fine -- that is the "
            "failure mode this gate exists to catch. Do not ship the candidate "
            "on the strength of a passing behavioural suite."
        )
        return 1

    print("VERDICT: PASS -- arms are numerically equivalent within tolerance.")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prompt-logprob perplexity parity gate.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--url", default="http://192.168.10.9:8100/v1",
                        help="OpenAI-compatible base URL (…/v1)")
    parser.add_argument("--model", default="deepseek-v4-flash-0731")
    parser.add_argument("--reps", type=int, default=2,
                        help="scoring repeats per passage; >1 also checks determinism")
    parser.add_argument("--timeout", type=int, default=600)
    parser.add_argument("--out", help="write results JSON here")
    parser.add_argument("--compare", nargs=2, metavar=("REF", "CAND"),
                        help="compare two result files and exit")
    parser.add_argument("--tolerance", type=float, default=1.0,
                        help="max allowed per-passage perplexity delta, percent")
    args = parser.parse_args()

    if args.compare:
        return compare(args.compare[0], args.compare[1], args.tolerance)

    print(f"scoring {len(CORPUS)} passages x {args.reps} reps against {args.url}")
    try:
        payload = run(args.url, args.model, args.reps, args.timeout)
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", "replace")[:400]
        print(f"HTTP {exc.code}: {detail}", file=sys.stderr)
        return 2

    agg = payload["aggregate"]
    print()
    print(f"aggregate perplexity : {agg['perplexity']:.4f}")
    print(f"aggregate mean NLL   : {agg['mean_nll']:.6f}")
    print(f"scored tokens        : {agg['scored_tokens']}")

    nondet = [p["passage"] for p in payload["passages"] if not p["deterministic"]]
    if nondet:
        print()
        print(f"WARNING: nondeterministic scoring on {', '.join(nondet)}.")
        print("Repeated teacher-forced scoring of identical text must be exact.")
        print("Investigate before using these numbers as a reference arm.")

    if args.out:
        with open(args.out, "w", encoding="utf-8") as handle:
            json.dump(payload, handle, indent=2)
        print(f"\nwrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
