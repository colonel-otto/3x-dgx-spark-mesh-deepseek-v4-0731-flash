#!/usr/bin/env bash
# eugr-sweep.sh — run one K-sweep point: the c grid against a live engine.
#
# Protocol (docs/ENGINE-AB-3NODE.md + BENCHMARK-METHODOLOGY.md):
#   - same harness as arm 1 (bench-miaai.py, 256-tok unique cold prefix)
#   - median-of->=5 trials on single-stream cells (33% noise band)
#   - exact 256-token completion windows (the arm-1 default of 128 remains
#     available for historical reproduction, but is not publishable here)
#   - WARMUP FIRST: arm-1's c=1 cell decayed 83.8 -> 57.8 tok/s across trials
#     under JIT compiles. Warm up, confirm the cute.compile miss counter has
#     frozen, and only then record.
#
# Usage: eugr-sweep.sh <nst> <mnbt> <engine-log> [outdir]
set -euo pipefail

NST="${1:?usage: eugr-sweep.sh <nst> <mnbt> <engine-log> [outdir]}"
MNBT="${2:?}"
ENGINE_LOG="${3:?}"
OUT="${4:-$HOME/eugr-sweep/nst${NST}-mnbt${MNBT}}"
OUTPUT_TOKENS="${OUTPUT_TOKENS:-256}"
METRICS_URL="${METRICS_URL:-http://127.0.0.1:8100/metrics}"
EXCLUSIVITY_SCRIPT="${EXCLUSIVITY_SCRIPT:-$HOME/exclusivity.py}"

BASE_URL="http://127.0.0.1:8100/v1"
MODEL="deepseek-v4-flash-eugr-ab"
HARNESS="$HOME/bench-miaai.py"
mkdir -p "$OUT"

[ -f "$ENGINE_LOG" ] || { echo "FATAL: engine log not found: $ENGINE_LOG" >&2; exit 1; }
[ -f "$HARNESS" ] || { echo "FATAL: harness not found: $HARNESS" >&2; exit 1; }
[ -f "$EXCLUSIVITY_SCRIPT" ] || {
  echo "FATAL: exclusivity gate not found: $EXCLUSIVITY_SCRIPT" >&2
  echo "       install scripts/exclusivity.py beside the harness or set EXCLUSIVITY_SCRIPT" >&2
  exit 1
}

echo "=== sweep point nst=$NST mnbt=$MNBT -> $OUT ==="

# --- warm the kernel caches, then verify the JIT miss counter has frozen -----
miss_before=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
echo "warmup: cute.compile disk-cache-miss count before = $miss_before"
for c in 1 4 8 16; do
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 2 --output-tokens "$OUTPUT_TOKENS" \
    > "$OUT/warmup-c${c}.log" 2>&1
done
sleep 5
miss_after=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
echo "warmup: cute.compile disk-cache-miss count after  = $miss_after"
echo "$miss_before -> $miss_after" > "$OUT/jit-miss-counter.txt"
if [ "$miss_before" != "$miss_after" ]; then
  echo "WARNING: JIT misses still growing during warmup ($miss_before -> $miss_after)." | tee -a "$OUT/jit-miss-counter.txt"
  echo "         Numbers below may still be contaminated; warming further." | tee -a "$OUT/jit-miss-counter.txt"
  for c in 1 16; do
    python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
      --prompt 256 --concurrency "$c" --repeat 3 --output-tokens "$OUTPUT_TOKENS" \
      >> "$OUT/warmup-extra.log" 2>&1
  done
  miss_final=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
  echo "after extra warmup = $miss_final" | tee -a "$OUT/jit-miss-counter.txt"
  if [ "$miss_final" != "$miss_after" ]; then
    echo "FATAL: JIT miss counter is still growing; refusing contaminated measurements." >&2
    exit 1
  fi
fi

miss_warm=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
echo "FROZEN before measurement: $miss_warm" | tee -a "$OUT/jit-miss-counter.txt"

# Exclusivity is a hard benchmark requirement. Warm-up traffic is deliberately
# before this baseline; only measured completions must appear in the delta.
idle_line=$(python3 "$EXCLUSIVITY_SCRIPT" --url "$METRICS_URL" --check-idle --timeout 60)
start_total=$(echo "$idle_line" | sed -n 's/^IDLE_OK start_request_success_total=\([0-9.]*\).*/\1/p')
[ -n "$start_total" ] || { echo "FATAL: could not capture exclusivity baseline" >&2; exit 1; }
printf 'metrics_url=%s\nstart_request_success_total=%s\n' "$METRICS_URL" "$start_total" \
  > "$OUT/exclusivity-start.txt"
expected_requests=0

# --- measure -----------------------------------------------------------------
: > "$OUT/rows.tsv"
printf 'nst\tmnbt\tc\toutput_tokens\tmedian_decode_tok_s\tagg_tok_s\tttft_ms\n' >> "$OUT/rows.tsv"
for c in 1 4 8 16; do
  echo "--- measuring c=$c"
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 5 --output-tokens "$OUTPUT_TOKENS" \
    > "$OUT/bench-c${c}.log" 2>&1

  [ "$(grep -c '^trial ' "$OUT/bench-c${c}.log")" -eq 5 ] || {
    echo "FATAL: c=$c did not produce exactly five successful trials" >&2
    exit 1
  }

  # FINAL line is the median-of-trials decode; agg/ttft are medians of the
  # per-trial values (the harness prints one line per trial).
  dec=$(grep '^FINAL' "$OUT/bench-c${c}.log" | sed -E 's/.*= ([0-9.]+) tok.*/\1/')
  agg=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*agg=([0-9.]+).*/\1/' \
        | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  ttft=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*ttft=([0-9]+)ms.*/\1/' \
        | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\t%s\n' \
    "$NST" "$MNBT" "$c" "$OUTPUT_TOKENS" "$dec" "$agg" "$ttft" >> "$OUT/rows.tsv"
  expected_requests=$((expected_requests + c * 5))
  echo "    c=$c decode=$dec agg=$agg ttft=${ttft}ms"
done

miss_end=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
echo "after measurement = $miss_end" | tee -a "$OUT/jit-miss-counter.txt"
[ "$miss_end" = "$miss_warm" ] || {
  echo "FATAL: JIT miss counter grew during measurement ($miss_warm -> $miss_end); results are void." >&2
  exit 1
}

python3 "$EXCLUSIVITY_SCRIPT" --url "$METRICS_URL" --verify \
  --start-total "$start_total" --expected "$expected_requests" \
  --out "$OUT/exclusivity.json"

# --- record the config that actually ran ------------------------------------
{
  echo "nst=$NST mnbt=$MNBT"
  echo "output_tokens=$OUTPUT_TOKENS"
  echo "engine_log=$ENGINE_LOG"
  grep -m1 'Initializing a V1 LLM engine' "$ENGINE_LOG" || true
  grep -m1 'GPU KV cache size' "$ENGINE_LOG" || true
  grep -m1 'max_num_scheduled_tokens' "$ENGINE_LOG" || true
  grep -m1 'virtual TP padding' "$ENGINE_LOG" || true
} > "$OUT/engine-config.txt" 2>&1

echo "=== done: $OUT/rows.tsv ==="
cat "$OUT/rows.tsv"
