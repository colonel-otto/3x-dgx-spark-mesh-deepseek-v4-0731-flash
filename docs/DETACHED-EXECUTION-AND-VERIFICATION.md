# Detached Execution and Verification Guide

This document establishes the operational standard for running long-running benchmarks, fabric gates, and quality suites under unstable network / SSH connections.

---

## 1. The Detached Execution Standard (`scripts/run_nohup.sh`)

When running multi-minute sweeps (e.g. 131K TTFT sweeps, GuideLLM matrices, or MTP concurrency evaluations), SSH drops will terminate the foreground process if not detached.

Use `scripts/run_nohup.sh` to run commands safely in the background with trapped PIDs, timestamped logs, and recorded exit codes.

### Usage

```bash
bash scripts/run_nohup.sh <command> [args...]
```

### Examples

```bash
# 1. Run the MTP concurrency sweep in the background
bash scripts/run_nohup.sh python3 scripts/benchmark_mtp_concurrency.py --mtp-k 2 --depth 8192 --out results/my_run.json

# 2. Run the full fabric gate
bash scripts/run_nohup.sh make gate-full CONFIG=configs/3spark-live.env

# 3. Run the logprob parity check
bash scripts/run_nohup.sh python3 scripts/logprob_parity.py
```

### Inspecting and Monitoring Runs

When launched, the script outputs paths for the log file, PID file, and exit code file under `results/nohup-runs/`:

```bash
# Live tail of the execution log
tail -f results/nohup-runs/<timestamp>_<command>.log

# Check if the process is still running
kill -0 $(cat results/nohup-runs/<timestamp>_<command>.pid) 2>/dev/null && echo "Running" || echo "Finished"

# Inspect the final return code
cat results/nohup-runs/<timestamp>_<command>.exit
```

---

## 2. Hardware Stability & Verification Protocol

### A. Persistent GPU Clock Locking (Eliminating Wake Jitter)

Run on all three nodes (`sparkmain`, `spark1`, `spark2`):

```bash
# Set persistence mode
sudo nvidia-smi -pm 1

# Disable auto-boost
sudo nvidia-smi --auto-boost-default=0

# Query maximum graphics clock
MAX_CLK=$(nvidia-smi --query-gpu=clocks.max.graphics --format=csv,noheader,nounits | head -n 1)

# Lock GPU clocks to prevent power-down state wake penalties (~22ms TTFT jitter)
sudo nvidia-smi -lgc ${MAX_CLK},${MAX_CLK}
```

### B. ConnectX-7 PCIe Gen5 x4 Verification

Verify that all 4 HCAs have negotiated full PCIe Gen5 x4 (32 GT/s):

```bash
lspci -vvv -s $(lspci | grep -i mell | awk '{print $1}' | head -n 1) | grep -E "LnkCap|LnkSta"
```

Expected output:
* `LnkCap: Port #0, Speed 32GT/s, Width x4`
* `LnkSta: Speed 32GT/s, Width x4`

---

## 3. Pre-Flight Verification Checklist

Before publishing any benchmark bundle:

1. **Fabric Gate**: `make gate-full CONFIG=configs/3spark-live.env` (Must pass 21/21 checks, 0 RDMA errors, `via NET/IB/x`).
2. **Correctness Check**: `python3 -c "import urllib.request, json; ..."` verifying `19 * 23 = 437`.
3. **Serving Parity Floor**: `python3 scripts/logprob_parity.py` ($\le 12.0\%$ noise floor tolerance).
4. **Window Assertion**: Forced 256 output tokens per request verified in the benchmark harness.