#!/usr/bin/env bash
# =============================================================================
# scripts/run_cluster_benchmark_suite.sh
# Comprehensive Long-Running Benchmark Suite for 3-Spark DSv4 Cluster
# =============================================================================
set -euo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)
PY="/home/sparkmain/bench_env/bin/python3"
if [[ ! -x "$PY" ]]; then
  PY="python3"
fi

TIMESTAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${TIMESTAMP}-locked-clocks-suite"
mkdir -p "$RUN_DIR"

echo "================================================================="
echo "=== 3-SPARK COMPREHENSIVE BENCHMARK SUITE STARTED ==="
echo "Timestamp : $TIMESTAMP"
echo "Run Dir   : $RUN_DIR"
echo "Python    : $PY"
echo "================================================================="

# 0. Assert cluster exclusivity & idle state (Issue #37)
echo "[0/5] Checking cluster idle & exclusivity pre-flight gate..."
$PY "$HERE/exclusivity.py" --check-idle --url "http://127.0.0.1:8100/metrics" --timeout 30.0

# 1. Snapshot hardware clocks & environment
echo "[1/5] Capturing environment & GPU clocks..."
nvidia-smi --query-gpu=name,clocks.current.graphics,clocks.max.graphics,temperature.gpu,power.draw --format=csv > "$RUN_DIR/gpu_clocks.csv" 2>&1 || true

# 2. Serving Parity / Determinism Check
echo "[2/5] Running logprob serving parity check..."
$PY "$HERE/logprob_parity.py" \
  --url "http://127.0.0.1:8100/v1" \
  --model "deepseek-v4-flash-0731" \
  --reps 5 \
  --tolerance 12.0 \
  --out "$RUN_DIR/logprob_parity.json" || { echo "WARNING: Parity check had errors" >&2; }

# 3. MTP Concurrency Sweep (K=2)
echo "[3/5] Running MTP K=2 concurrency sweep (cc=1,4,8,16 at 8K depth)..."
$PY "$HERE/benchmark_mtp_concurrency.py" \
  --url "http://127.0.0.1:8100/v1" \
  --metrics-url "http://127.0.0.1:8100/metrics" \
  --model "deepseek-v4-flash-0731" \
  --mtp-k 2 \
  --depth 8192 \
  --concurrencies "1,4,8,16" \
  --reps 3 \
  --max-tokens 256 \
  --out "$RUN_DIR/mtp_k2_concurrency.json"

# 4. Prefill TTFT & Decode Depth Sweep (2K - 131K)
echo "[4/5] Running prefill TTFT & decode depth sweep (2K-131K, 256 tokens)..."
$PY "$HERE/benchmark_prefill_depth.py" \
  --url "http://127.0.0.1:8100/v1" \
  --model "deepseek-v4-flash-0731" \
  --batched-tokens 8192 \
  --depths "2048,8192,32768,65536,131072" \
  --reps 3 \
  --max-tokens 256 \
  --out "$RUN_DIR/prefill_depth.json"

# 5. Multi-Turn APC Warm-Path Benchmark
echo "[5/5] Running multi-turn Automatic Prefix Caching (APC) warm-path benchmark..."
$PY "$HERE/multiturn_apc.py" \
  --base-url "http://127.0.0.1:8100/v1" \
  --model "deepseek-v4-flash-0731" \
  --depths "8192,32768,131072" \
  --turns 5 \
  --sessions 2 \
  --max-tokens 256 \
  --gap-s 5 \
  --label "locked-clocks-apc" \
  --output "$RUN_DIR/apc_warm_path.json"

echo "================================================================="
echo "=== ALL BENCHMARKS COMPLETED SUCCESSFULLY ==="
echo "Results saved in: $RUN_DIR"
echo "================================================================="