#!/usr/bin/env python3
"""OpenAI-compatible streaming benchmark for before/after DGX Spark tests."""
from __future__ import annotations
import argparse, concurrent.futures, json, time
from dataclasses import dataclass, asdict
from pathlib import Path
import requests

WORDS = "amber cedar delta ember fjord granite harbor iris juniper kestrel lumen maple nova orbit pine quartz river solar tundra umber vector willow xenon yarrow zenith".split()
NEEDLE_VALUE = "ORBITAL-CEDAR-9417"


def make_prompt(target_tokens: int, needle_position: float = 0.5) -> str:
    # Approximate construction only; actual prompt token count is taken from server usage.
    target_words = max(64, int(target_tokens * 0.72))
    parts = [WORDS[i % len(WORDS)] for i in range(target_words)]
    idx = min(len(parts)-1, max(0, int(len(parts) * needle_position)))
    parts[idx] = f"DGX_NEEDLE={NEEDLE_VALUE}"
    body = " ".join(parts)
    return (
        "Read the entire context. A single marker named DGX_NEEDLE appears inside it. "
        "At the end, answer with only the value after DGX_NEEDLE= and nothing else.\n\n"
        + body
        + "\n\nWhat is the exact DGX_NEEDLE value?"
    )


def discover_model(api_base: str, timeout: float) -> str:
    r = requests.get(api_base.rstrip("/") + "/v1/models", timeout=timeout)
    r.raise_for_status()
    data = r.json().get("data", [])
    if not data:
        raise RuntimeError("/v1/models returned no models")
    return data[0]["id"]


@dataclass
class Result:
    label: str
    context_target: int
    concurrency: int
    request_index: int
    needle_position: float
    ok: bool
    status_code: int | None
    ttft_s: float | None
    e2e_s: float
    prompt_tokens: int | None
    output_tokens: int | None
    decode_tps: float | None
    needle_correct: bool
    output_preview: str
    error: str | None


def one_request(api_base: str, model: str, label: str, context: int, concurrency: int,
                request_index: int, max_tokens: int, timeout: float, needle_position: float) -> Result:
    url = api_base.rstrip("/") + "/v1/chat/completions"
    payload = {
        "model": model,
        "messages": [{"role": "user", "content": make_prompt(context, needle_position)}],
        "temperature": 0,
        "max_tokens": max_tokens,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    start = time.perf_counter()
    first = None
    text = []
    usage = {}
    status = None
    try:
        with requests.post(url, json=payload, stream=True, timeout=(20, timeout)) as r:
            status = r.status_code
            r.raise_for_status()
            for raw in r.iter_lines(decode_unicode=True):
                if not raw or not raw.startswith("data:"):
                    continue
                data = raw[5:].strip()
                if data == "[DONE]":
                    break
                obj = json.loads(data)
                if obj.get("usage"):
                    usage = obj["usage"]
                for choice in obj.get("choices", []):
                    delta = choice.get("delta") or {}
                    reasoning = delta.get("reasoning_content") or ""
                    content = delta.get("content") or ""
                    emitted = reasoning or content
                    if emitted and first is None:
                        first = time.perf_counter()
                    if content:
                        text.append(content)
        end = time.perf_counter()
        output = "".join(text).strip()
        out_tokens = usage.get("completion_tokens")
        prompt_tokens = usage.get("prompt_tokens")
        ttft = (first - start) if first else None
        decode_elapsed = (end - first) if first else None
        decode_tokens = max(0, out_tokens - 1) if isinstance(out_tokens, int) else None
        decode_tps = (decode_tokens / decode_elapsed) if decode_tokens is not None and decode_elapsed and decode_elapsed > 0 else None
        return Result(label, context, concurrency, request_index, needle_position, True, status,
                      ttft, end-start, prompt_tokens, out_tokens, decode_tps,
                      NEEDLE_VALUE in output, output[:240], None)
    except Exception as exc:
        end = time.perf_counter()
        return Result(label, context, concurrency, request_index, needle_position, False, status,
                      None, end-start, None, None, None, False, "", repr(exc))


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--api-base", required=True)
    ap.add_argument("--model", default="auto")
    ap.add_argument("--label", required=True)
    ap.add_argument("--contexts", default="2048,8192,32768")
    ap.add_argument("--concurrencies", default="1,3,6")
    ap.add_argument("--max-tokens", type=int, default=256)
    ap.add_argument("--repetitions", type=int, default=3)
    ap.add_argument("--timeout", type=float, default=900)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    model = discover_model(args.api_base, 30) if args.model == "auto" else args.model
    contexts = [int(x) for x in args.contexts.split(",") if x]
    concurrencies = [int(x) for x in args.concurrencies.split(",") if x]
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    all_results: list[Result] = []
    with output_path.open("w", encoding="utf-8") as fh:
        for context in contexts:
            for concurrency in concurrencies:
                for rep in range(args.repetitions):
                    positions = [0.1, 0.5, 0.9]
                    wave_start = time.perf_counter()
                    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
                        futs = [pool.submit(one_request, args.api_base, model, args.label, context,
                                            concurrency, i, args.max_tokens, args.timeout,
                                            positions[(rep + i) % len(positions)])
                                for i in range(concurrency)]
                        wave = [f.result() for f in futs]
                    wall = time.perf_counter() - wave_start
                    total_out = sum(r.output_tokens or 0 for r in wave)
                    aggregate_tps = (total_out / wall) if wall > 0 else None
                    for r in wave:
                        row = asdict(r)
                        row["wave_wall_s"] = wall
                        row["wave_aggregate_output_tps"] = aggregate_tps
                        row["model"] = model
                        fh.write(json.dumps(row, sort_keys=True) + "\n")
                        fh.flush()
                        all_results.append(r)
                    print(f"context={context} concurrency={concurrency} rep={rep+1} "
                          f"ok={sum(r.ok for r in wave)}/{len(wave)} aggregate_tps={aggregate_tps}")

    failures = [r for r in all_results if not r.ok]
    incorrect = [r for r in all_results if r.ok and not r.needle_correct]
    print(f"wrote {len(all_results)} requests to {output_path}")
    print(f"request_failures={len(failures)} needle_failures={len(incorrect)}")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
