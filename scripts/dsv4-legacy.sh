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
# ⚠️ A parked container is NOT the durable copy: on 2026-08-30 all three were
# docker-rm'd within minutes by concurrent leaked-container cleanup during an
# engine sweep. The `save` tarball on sparkmain is the store no docker-level
# hygiene can touch; the parked containers are the convenience layer. The
# keep=parked-legacy label marks them deliberate -- cleanup tooling must skip
# containers carrying it.
#
# Usage:
#   scripts/dsv4-legacy.sh create           # park on all nodes (idempotent)
#   scripts/dsv4-legacy.sh status           # where is it parked / running?
#   scripts/dsv4-legacy.sh shell <node>     # start it + drop into bash
#   scripts/dsv4-legacy.sh stop             # stop after maintenance (keeps it parked)
#   scripts/dsv4-legacy.sh rm               # unpark everywhere (image stays)
#   scripts/dsv4-legacy.sh save             # durable tarball -> sparkmain:~/images/
#   scripts/dsv4-legacy.sh load             # restore image from that tarball
set -euo pipefail

NODES=(sparkmain spark1 spark-sep)
IMAGE=dsv4-3spark:0.1.1
NAME=dsv4-legacy
TARBALL='$HOME/images/dsv4-3spark-0.1.1.tar'

# ⛔ The image is a docker-commit of a LIVE COMPOSE CONTAINER, so it bakes in
# com.docker.compose.project=dspark-vllm-gx10 -- and every container created
# from it inherits that label, which makes `docker compose -p dspark-vllm-gx10
# down` (the dsv4 stop path, run nightly by the sweep) reap it as its own.
# That is what killed the first two batches. The --label overrides below
# rewrite the compose identity so no compose teardown can ever match it.
create_cmd="docker inspect $NAME >/dev/null 2>&1 && echo 'already parked' || \
docker create --name $NAME \
  --label keep=parked-legacy \
  --label com.docker.compose.project=dsv4-legacy-parked \
  --label com.docker.compose.service=dsv4-legacy-parked \
  --label com.docker.compose.oneoff=True \
  --label info='parked legacy runtime, deliberately stopped -- do not reap; see 3spark-dsv4/scripts/dsv4-legacy.sh' \
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
  save)
    ssh sparkmain "mkdir -p ~/images && ls -la $TARBALL 2>/dev/null && echo 'tarball already exists' || \
      { nohup docker save -o $TARBALL $IMAGE >/dev/null 2>&1 && ls -la $TARBALL; }"
    ;;
  load)
    ssh sparkmain "docker load -i $TARBALL"
    ;;
  *) echo "usage: $0 {create|status|shell <node>|stop|rm|save|load}" >&2; exit 2 ;;
esac
