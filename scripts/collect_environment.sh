#!/usr/bin/env bash
set -euo pipefail
HERE=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
source "$HERE/common.sh"
load_config "${1:?usage: collect_environment.sh CONFIG OUTDIR}"
OUT=${2:?output directory required}
mkdir -p "$OUT/environment"

cat > "$OUT/manifest.env" <<MANIFEST
captured_utc=$(date -u +%FT%TZ)
run_label=${RUN_LABEL}
topology=${TOPOLOGY:-unknown}
node_count=${#NODES[@]}
model=${MODEL:-unknown}
model_revision=${MODEL_REVISION:-unknown}
tp_size=${TP_SIZE:-unknown}
pp_size=${PP_SIZE:-unknown}
pp_layer_partition=${PP_LAYER_PARTITION:-auto}
distributed_backend=${DISTRIBUTED_BACKEND:-unknown}
speculative=${SPECULATIVE:-unknown}
vllm_image=${VLLM_IMAGE:-unknown}
api_base=${API_BASE:-unknown}
nvidia_playbook_ref=${NVIDIA_PLAYBOOK_REF:-unknown}
MANIFEST

for ip in "${NODES[@]}"; do
  safe=${ip//[:.]/_}
  ssh_node "$ip" "
    echo '## hostname'; hostname
    echo '## date'; date -u +%FT%TZ
    echo '## dgx-release'; cat /etc/dgx-release 2>/dev/null || true
    echo '## os-release'; cat /etc/os-release
    echo '## uname'; uname -a
    echo '## nvidia-smi'; nvidia-smi
    echo '## free'; free -b
    echo '## ibdev2netdev'; command -v ibdev2netdev >/dev/null && ibdev2netdev || true
    echo '## ip'; ip -br addr
    echo '## docker'; command -v docker >/dev/null && docker --version || true
    echo '## docker-images'; command -v docker >/dev/null && docker images --digests --format '{{.Repository}} {{.Tag}} {{.Digest}} {{.ID}}' || true
    echo '## python'; command -v python3 >/dev/null && python3 --version || true
    echo '## torch'; command -v python3 >/dev/null && python3 -c 'import torch; print(torch.__version__, torch.version.cuda)' 2>/dev/null || true
    echo '## vllm'; command -v vllm >/dev/null && vllm --version || true
    echo '## ray'; command -v ray >/dev/null && ray --version || true
  " > "$OUT/environment/${safe}.txt" 2>&1
done
