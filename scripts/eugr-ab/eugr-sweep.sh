#!/usr/bin/env bash
# eugr-sweep.sh — run one K-sweep point: the c grid against a live engine.
#
# Protocol (docs/ENGINE-AB-3NODE.md + BENCHMARK-METHODOLOGY.md):
#   - same harness as arm 1 (bench-miaai.py, 256-tok unique cold prefix)
#   - median-of->=5 trials on single-stream cells (33% noise band)
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

BASE_URL="http://127.0.0.1:8100/v1"
MODEL="deepseek-v4-flash-eugr-ab"
HARNESS="$HOME/bench-miaai.py"
mkdir -p "$OUT"

echo "=== sweep point nst=$NST mnbt=$MNBT -> $OUT ==="

# --- warm the kernel caches, then verify the JIT miss counter has frozen -----
miss_before=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
echo "warmup: cute.compile disk-cache-miss count before = $miss_before"
for c in 1 4 8 16; do
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 2 > "$OUT/warmup-c${c}.log" 2>&1 || true
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
      --prompt 256 --concurrency "$c" --repeat 3 >> "$OUT/warmup-extra.log" 2>&1 || true
  done
  miss_final=$(grep -c 'disk-cache-miss' "$ENGINE_LOG" || true)
  echo "after extra warmup = $miss_final" | tee -a "$OUT/jit-miss-counter.txt"
fi

# --- measure -----------------------------------------------------------------
: > "$OUT/rows.tsv"
printf 'nst\tmnbt\tc\tmedian_decode_tok_s\tagg_tok_s\tttft_ms\n' >> "$OUT/rows.tsv"
for c in 1 4 8 16; do
  echo "--- measuring c=$c"
  python3 "$HARNESS" --base-url "$BASE_URL" --model "$MODEL" \
    --prompt 256 --concurrency "$c" --repeat 5 > "$OUT/bench-c${c}.log" 2>&1

  # FINAL line is the median-of-trials decode; agg/ttft are medians of the
  # per-trial values (the harness prints one line per trial).
  dec=$(grep '^FINAL' "$OUT/bench-c${c}.log" | sed -E 's/.*= ([0-9.]+) tok.*/\1/')
  agg=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*agg=([0-9.]+).*/\1/' \
        | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  ttft=$(grep '^trial' "$OUT/bench-c${c}.log" | sed -E 's/.*ttft=([0-9]+)ms.*/\1/' \
        | sort -n | awk '{v[NR]=$1} END{print (NR%2)?v[(NR+1)/2]:(v[NR/2]+v[NR/2+1])/2}')
  printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$NST" "$MNBT" "$c" "$dec" "$agg" "$ttft" >> "$OUT/rows.tsv"
  echo "    c=$c decode=$dec agg=$agg ttft=${ttft}ms"
done

# --- record the config that actually ran ------------------------------------
{
  echo "nst=$NST mnbt=$MNBT"
  echo "engine_log=$ENGINE_LOG"
  grep -m1 'Initializing a V1 LLM engine' "$ENGINE_LOG" || true
  grep -m1 'GPU KV cache size' "$ENGINE_LOG" || true
  grep -m1 'max_num_scheduled_tokens' "$ENGINE_LOG" || true
  grep -m1 'virtual TP padding' "$ENGINE_LOG" || true
} > "$OUT/engine-config.txt" 2>&1

echo "=== done: $OUT/rows.tsv ==="
cat "$OUT/rows.tsv"
