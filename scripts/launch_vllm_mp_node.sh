#!/usr/bin/env bash
set -euo pipefail
# Launch one vLLM process for a multi-node multiprocessing deployment.
# Run once on EACH Spark with rank 0, 1, and 2 respectively.
CONFIG=${1:?usage: launch_vllm_mp_node.sh CONFIG NODE_RANK}
RANK=${2:?node rank required: 0, 1, or 2}
set -a; source "$CONFIG"; set +a
: "${MODEL:?MODEL required}"
: "${TP_SIZE:=1}"
: "${PP_SIZE:=3}"
: "${PP_LAYER_PARTITION:=14,15,14}"
: "${MASTER_ADDR:?MASTER_ADDR required in config}"
: "${MASTER_PORT:=29501}"
: "${NNODES:=3}"

if [[ "${SPECULATIVE:-false}" != false ]]; then
  echo 'Refusing initial candidate launch: SPECULATIVE must be false.' >&2
  exit 2
fi
if [[ "$TP_SIZE" != 1 || "$PP_SIZE" != 3 || "$NNODES" != 3 ]]; then
  echo "Expected TP=1 PP=3 NNODES=3; got TP=$TP_SIZE PP=$PP_SIZE NNODES=$NNODES" >&2
  exit 2
fi
if ! [[ "$RANK" =~ ^[0-2]$ ]]; then
  echo "NODE_RANK must be 0, 1, or 2; got $RANK" >&2
  exit 2
fi

export VLLM_PP_LAYER_PARTITION="$PP_LAYER_PARTITION"
export NCCL_IB_SUBNET_AWARE_ROUTING=1
export NCCL_NET_PLUGIN=none
if [[ -n "${MGMT_IFNAME:-}" ]]; then
  export NCCL_SOCKET_IFNAME="$MGMT_IFNAME"
  export GLOO_SOCKET_IFNAME="$MGMT_IFNAME"
fi

cmd=(vllm serve "$MODEL"
  --distributed-executor-backend mp
  --tensor-parallel-size "$TP_SIZE"
  --pipeline-parallel-size "$PP_SIZE"
  --nnodes "$NNODES"
  --node-rank "$RANK"
  --master-addr "$MASTER_ADDR"
  --master-port "$MASTER_PORT")

if [[ "$RANK" != 0 ]]; then
  cmd+=(--headless)
fi
if [[ -n "${MODEL_REVISION:-}" ]]; then
  cmd+=(--revision "$MODEL_REVISION")
fi
if [[ -n "${EXTRA_VLLM_ARGS:-}" ]]; then
  # shellcheck disable=SC2206
  extra=( ${EXTRA_VLLM_ARGS} )
  cmd+=("${extra[@]}")
fi
printf 'Launching rank %s:' "$RANK"; printf ' %q' "${cmd[@]}"; printf '\n'
exec "${cmd[@]}"
