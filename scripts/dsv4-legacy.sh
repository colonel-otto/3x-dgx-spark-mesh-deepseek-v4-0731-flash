#!/usr/bin/env bash
# Park the retired dsv4-3spark:0.1.1 runtime as a STOPPED maintenance
# container on every node, so the image stays instantly startable after the
# fleet moves to the upstream-built image.
#
# Why a created container instead of just the image tag: a container pins its
# image against `docker image prune -a`, and a never-started container costs
# only metadata. It does NOT touch the live dsv4 stack -- different container
# name, and it runs `sleep infinity`, never `vllm serve`.
#
# Usage:
#   scripts/dsv4-legacy.sh create           # park on all nodes (idempotent)
#   scripts/dsv4-legacy.sh status           # where is it parked / running?
#   scripts/dsv4-legacy.sh shell <node>     # start it + drop into bash
#   scripts/dsv4-legacy.sh stop             # stop after maintenance (keeps it parked)
#   scripts/dsv4-legacy.sh rm               # unpark everywhere (image stays)
set -euo pipefail

NODES=(sparkmain spark1 spark-sep)
IMAGE=dsv4-3spark:0.1.1
NAME=dsv4-legacy

create_cmd="docker inspect $NAME >/dev/null 2>&1 && echo 'already parked' || \
docker create --name $NAME \
  --gpus all --network host --ipc host --shm-size 64gb \
  -v \$HOME/.cache/huggingface:/cache/huggingface \
  -v \$HOME/models/dsv4-flash-dspark-abliterated:/models/dsv4-abliterated:ro \
  --entrypoint sleep $IMAGE infinity"

each() { for n in "${NODES[@]}"; do echo "== $n =="; ssh "$n" "$1"; done; }

case "${1:-status}" in
  create) each "$create_cmd" ;;
  status) each "docker ps -a --filter name=$NAME --format '{{.Names}}  {{.Image}}  {{.Status}}'" ;;
  stop)   each "docker stop -t 2 $NAME >/dev/null 2>&1 && echo stopped || echo 'not running'" ;;
  rm)     each "docker rm -f $NAME >/dev/null 2>&1 && echo unparked || echo 'not present'" ;;
  shell)
    node=${2:?usage: dsv4-legacy.sh shell <node>}
    ssh -t "$node" "docker start $NAME >/dev/null && docker exec -it $NAME bash"
    ;;
  *) echo "usage: $0 {create|status|shell <node>|stop|rm}" >&2; exit 2 ;;
esac
