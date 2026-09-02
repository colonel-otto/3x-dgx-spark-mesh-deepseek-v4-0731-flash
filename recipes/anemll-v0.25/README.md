# Recipe: Anemll v0.25.1 + DSpark Hotfixes (`anemll-v0.25`)

This recipe packages the baseline DeepSeek-V4 Flash deployment lineage across the 3x DGX Spark mesh cluster.

---

## Overview & Architecture

* **Base Image:** `ghcr.io/anemll/dspark-vllm-gx10:0.1.1`
* **vLLM Engine Version:** `v0.25.1`
* **TP Strategy:** Attention-group padding patch (`64 -> 72` heads, `8 -> 9` attention groups) enabling exact mathematical correctness at TP=3 over switchless RoCE ring interconnects.
* **Speculative Decoding:** MTP $K=2$ speculative heads.
* **KV Cache Dtype:** `nvfp4_ds_mla`

---

## Directory Contents

* **`Dockerfile.runtime`**: Builds the container overlay by applying `apply_tp3_patch.py`, `hotfix-dsv4-issue26-hybrid-swa-min.py`, and `hotfix-dsv4-issue27-partial-prefill-concurrency.py`.
* **`docker-compose.yml`**: Compose orchestration definition configured with host networking, pinned fabric names, and shared memory parameters.
* **`env/`**: Cluster environment configurations for 2-node and 3-node setups:
  * `2spark-live.env` & `3spark-live.env` (live node configurations)
  * `tp3.env.example` & per-node `.env.example` files
* **`patches/`**: Python patch scripts and test fixtures for TP=3 attention grouping and issue hotfixes.
* **`scripts/`**: Bootstrap, launch, and fabric-gating scripts (`bootstrap_nccl.sh`, `cluster_tp2.sh`, `fabric_gate.sh`, `launch_vllm_candidate.sh`, `launch_vllm_mp_node.sh`).

---

## Quick Start

### 1. Build Overlay Container
```bash
docker build -t dsv4-3spark:0.1.1 -f Dockerfile.runtime .
```

### 2. Verify Attention-Group Padding
```bash
python3 patches/apply_tp3_patch.py --check
```

### 3. Launch Multi-Node Cluster
```bash
# Verify RoCE fabric readiness
bash scripts/fabric_gate.sh

# Launch via docker-compose using the target environment
docker compose --env-file env/3spark-live.env up -d
```
