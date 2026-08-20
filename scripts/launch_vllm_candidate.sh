#!/usr/bin/env bash
set -euo pipefail
# This launches only the vLLM server. Start/verify the 3-node Ray cluster first.
# Keep EXTRA_VLLM_ARGS identical to the working baseline except for TP/PP.
CONFIG=${1:?usage: launch_vllm_candidate.sh CONFIG}
set -a; source "$CONFIG"; set +a
: "${MODEL:?MODEL required}"
: "${TP_SIZE:=1}"
: "${PP_SIZE:=3}"
: "${PP_LAYER_PARTITION:=14,15,14}"

if [[ "${SPECULATIVE:-false}" != false ]]; then
  echo 'Refusing initial candidate launch: SPECULATIVE must be false.' >&2
  exit 2
fi
if [[ "$TP_SIZE" != 1 || "$PP_SIZE" != 3 ]]; then
  echo "Expected candidate TP_SIZE=1 PP_SIZE=3; got TP=$TP_SIZE PP=$PP_SIZE" >&2
  exit 2
fi

export VLLM_PP_LAYER_PARTITION="$PP_LAYER_PARTITION"
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none

cmd=(vllm serve "$MODEL"
  --distributed-executor-backend ray
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size "$PP_SIZE")
if [[ -n "${MODEL_REVISION:-}" ]]; then
  cmd+=(--revision "$MODEL_REVISION")
fi
# EXTRA_VLLM_ARGS is intentionally shell-split so existing flags can be copied in.
# Do not place secrets in it.
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_VLLM_ARGS} )
  cmd+=("${extra[@]}")
fi
printf 'Launching:'; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
