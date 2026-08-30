#!/usr/bin/env bash
# rerun_issue22_deep_decode.sh — re-measure the issue #22 exemption at the 1M ceiling.
#
# WHY THIS EXISTS: our exemption from MiaAI issue #22 (nvfp4_ds_mla routed to the
# slow BF16 kernel, ~1 tok/s collapse at 600K+ context) was measured under
# max_model_len=460800 and never went past 409,600 prompt tokens. We now ship
# MAX_MODEL_LEN=1048576, so the regime where the bug bites is reachable and
# unmeasured. See docs/ISSUE22-EXEMPTION-STALE.md.
#
# WHAT IT MEASURES: single-stream decode tok/s at prompt depths
#   262144 / 409600 / 524288 / 716800 / 870400 / 1046528 (~1M, leaves headroom
#   for the 256-token window under the 1,048,576 ceiling),
# n>=5 reps per cell (default 7), 2 warmups per shape, asserted 256-token decode
# windows (min_tokens == max_tokens == 256, ignore_eos, per-rep
# completion_tokens assertion — the harness FAILS a rep rather than publish a
# collapsed window), prefix-cache misses verified per rep. All of that is
# inherited from scripts/decode_depth_sweep.py; this wrapper adds the gates and
# the bundle.
#
# DECISION RULE (docs/ISSUE22-EXEMPTION-STALE.md): flat-in-family through ~1M
# renews the exemption; a collapse toward ~1 tok/s anywhere in 524K–1M means
# issue #22 bites and hotfix-nvfp4-ds-mla-issue22.sh gets a matched A/B.
#
# REFUSES TO RUN if the engine does not answer — this script never starts,
# stops, or restarts anything. Recovery is the operator's job.
#
# BENCHMARK-POLICY compliance (docs/BENCHMARK-POLICY.md):
#   Req 1 (fabric gate): run scripts/fabric_gate.sh yourself before this script
#          and drop fabric-gate.json into the bundle; the run README records
#          PRESENT or ABSENT. This wrapper does not run the gate because the
#          gate's NCCL arm needs the engine STOPPED, and by design this script
#          only runs with the engine UP — gate first, boot, then measure.
#   Req 2 (window): 256-token asserted windows, via decode_depth_sweep.py.
#   Req 3 (spread): per-rep JSONL + summary with min/max/spread committed.
#   Req 4 (live config): captures /v1/models + engine /metrics config lines;
#          capture `ps -eo args` on the head yourself if SSH is available.
#   Req 5 (exclusivity): asserts idle before, verifies request-count delta
#          after, via scripts/exclusivity.py; FAILS on foreign traffic.
#
# Usage (from the repo root, this PC or the head node):
#   bash scripts/rerun_issue22_deep_decode.sh
#   BASE_URL=http://<head-mgmt-ip>:8100 bash scripts/rerun_issue22_deep_decode.sh  # from this PC
#   REPS=5 DEPTHS="262144 524288" bash scripts/rerun_issue22_deep_decode.sh    # partial rerun
#
# Long runs: a 1M prefill alone is many minutes. Run under nohup+tee per
# docs/DETACHED-EXECUTION-AND-VERIFICATION.md so a dropped terminal does not
# orphan the sweep:
#   nohup bash scripts/rerun_issue22_deep_decode.sh > /tmp/issue22-rerun.log 2>&1 &
set -uo pipefail

HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
ROOT=$(cd "$HERE/.." && pwd)

BASE_URL="${BASE_URL:-http://127.0.0.1:8100}"   # head-local; from another box: BASE_URL=http://<head-mgmt-ip>:8100
API="$BASE_URL/v1"
METRICS_URL="$BASE_URL/metrics"
MODEL="${MODEL:-deepseek-v4-flash-0731}"
# 1046528 = 1048576 - 2048: keeps prompt + 256-token window + tokenizer
# estimation error safely under the ceiling (build_prompt targets, server-side
# usage reports the actual count, which is what gets published).
DEPTHS="${DEPTHS:-262144 409600 524288 716800 870400 1046528}"
REPS="${REPS:-7}"            # policy floor is n>=5
WARMUPS="${WARMUPS:-2}"

STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN_DIR="$ROOT/results/${STAMP}-issue22-deep-decode"

log() { echo "[$(date -Iseconds)] $*"; }
die() { log "FATAL: $*"; exit 1; }

# ---- HARD GATE: the engine must already be up. We never start it. ----------
log "checking engine at $API/models ..."
MODELS_JSON=$(curl -sf -m 10 "$API/models") || die \
"engine is DOWN: $API/models did not answer.
This script measures a RUNNING engine and will not start one.
Bring the cluster up first (operator action), verify with:
    curl -s $API/models
then re-run."
echo "$MODELS_JSON" | grep -q "$MODEL" || die \
"engine answered but does not serve '$MODEL'. Got: $MODELS_JSON"
log "engine is up and serves $MODEL"

# Reject a stale ceiling: the whole point is measuring above 460800.
MAXLEN=$(echo "$MODELS_JSON" | python3 -c 'import json,sys
d=json.load(sys.stdin)
print(max((m.get("max_model_len") or 0) for m in d.get("data",[])))' 2>/dev/null || echo 0)
if [ "${MAXLEN:-0}" -gt 0 ] && [ "$MAXLEN" -lt 1048576 ]; then
  die "engine max_model_len=$MAXLEN < 1048576 — this is not the 1M config this rerun exists to test."
fi

mkdir -p "$RUN_DIR"

# ---- Req 4: capture what the LIVE engine says about itself -----------------
echo "$MODELS_JSON" | python3 -m json.tool > "$RUN_DIR/v1-models.json" 2>/dev/null || \
  echo "$MODELS_JSON" > "$RUN_DIR/v1-models.json"
curl -sf -m 10 "$METRICS_URL" | grep -E "cache_config_info|vllm:spec_decode" \
  > "$RUN_DIR/engine-metrics-config.txt" || true
log "NOTE (Req 4): also capture 'ps -eo args | grep vllm' on the head into $RUN_DIR/engine-config.txt"

# ---- Req 5: exclusivity — assert idle, remember the success counter --------
# exclusivity.py prints "IDLE_OK start_request_success_total=<N>"; keep <N>.
IDLE_LINE=$(python3 "$HERE/exclusivity.py" --url "$METRICS_URL" --check-idle --timeout 60) \
  || die "exclusivity pre-check failed: engine is not idle (foreign traffic?). Run is VOID before it starts."
START_TOTAL=${IDLE_LINE##*=}
case "$START_TOTAL" in (''|*[!0-9.]*) die "could not parse start counter from: $IDLE_LINE";; esac
log "cluster idle; request_success_total start=$START_TOTAL"

# ---- The sweep -------------------------------------------------------------
DEPTH_CSV=$(echo "$DEPTHS" | tr ' ' ',')
N_DEPTHS=$(echo "$DEPTHS" | wc -w)
EXPECTED=$(( N_DEPTHS * (REPS + WARMUPS) ))
log "sweep: depths [$DEPTH_CSV] x (${WARMUPS} warmups + ${REPS} reps) -> $RUN_DIR"

python3 "$HERE/decode_depth_sweep.py" \
  --base-url "$API" \
  --model "$MODEL" \
  --depths "$DEPTH_CSV" \
  --max-tokens 256 \
  --warmups "$WARMUPS" \
  --reps "$REPS" \
  --label issue22-rerun \
  --output "$RUN_DIR/issue22-deep-decode.jsonl"
SWEEP_RC=$?

# ---- Req 5: verify nothing else was served during the window ---------------
# Failed/aborted reps make EXPECTED an upper bound; exclusivity.py compares the
# counter delta and writes the observed values either way.
python3 "$HERE/exclusivity.py" --url "$METRICS_URL" --verify \
  --start-total "$START_TOTAL" --expected "$EXPECTED" \
  --out "$RUN_DIR/exclusivity.json" \
  || log "WARNING: exclusivity verify failed — inspect $RUN_DIR/exclusivity.json; foreign traffic voids the run."

# ---- Bundle README ---------------------------------------------------------
GATE_STATE="ABSENT"
[ -f "$RUN_DIR/fabric-gate.json" ] && GATE_STATE="PRESENT"
cat > "$RUN_DIR/README.md" <<EOF
# Issue #22 re-measure — deep decode at the 1M ceiling — $STAMP

**Status:** $( [ "$SWEEP_RC" -eq 0 ] && echo 'CURRENT' || echo 'INCOMPLETE (sweep rc='"$SWEEP_RC"')' ) • **Nodes:** 3 • **TP:** 3 • **Fabric gate:** \`$GATE_STATE\`

Re-measures the MiaAI issue #22 exemption (nvfp4_ds_mla -> slow BF16 kernel,
~1 tok/s at 600K+) which was previously cleared only up to 409,600 prompt
tokens under max_model_len=460800. See docs/ISSUE22-EXEMPTION-STALE.md for the
decision rule this run feeds.

- Harness: scripts/rerun_issue22_deep_decode.sh -> scripts/decode_depth_sweep.py
- Depths (target tokens): $DEPTHS
- Reps per cell: $REPS (+$WARMUPS warmups per shape, discarded)
- Decode window: 256 tokens, pinned (min_tokens+ignore_eos) and asserted per rep
- Prefix cache: unique per-rep header; cached_tokens checked per rep
- Prompt generator: decode_depth_sweep.py build_prompt (deterministic filler,
  salt "issue22-rerun-{w|r}N-DEPTH")
- Engine self-report: v1-models.json, engine-metrics-config.txt
- Exclusivity: exclusivity.json (Req 5)
- Fabric gate: $( [ "$GATE_STATE" = PRESENT ] && echo 'fabric-gate.json (run before boot)' || echo 'NOT CAPTURED — run scripts/fabric_gate.sh before the boot next time; without it this bundle is fabric_gate: ABSENT in results/INDEX.md' )

## Files
- issue22-deep-decode.jsonl — per-rep records (decode_tok_s, ttft_s, prompt/cached/completion tokens)
- issue22-deep-decode-summary.json — per-depth median/min/max/spread/cache_hits

## Interpretation
Flat-in-family decode through ~1M renews the exemption at the new ceiling.
A collapse toward ~1 tok/s in any 524K–1M cell means issue #22 bites us:
next step is a matched A/B of MiaAI's hotfix-nvfp4-ds-mla-issue22.sh per
docs/BENCHMARK-POLICY.md.
EOF

log "bundle: $RUN_DIR"
[ "$SWEEP_RC" -eq 0 ] || die "sweep exited rc=$SWEEP_RC — bundle is INCOMPLETE, do not publish as-is."
log "done. Next: generate/refresh results/INDEX.md and update docs/ISSUE22-EXEMPTION-STALE.md with the verdict."
