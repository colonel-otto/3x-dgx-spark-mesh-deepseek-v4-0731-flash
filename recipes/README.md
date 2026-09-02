# DeepSeek-V4 Flash Engine Recipes (3x DGX Spark Mesh)

This directory organizes the inference runtime engine recipes evaluated across the 3x DGX Spark switchless RoCE mesh cluster.

Each recipe encapsulates the container image provenance, environment configuration, tensor parallel sharding strategies, kernel hotfixes, and orchestration scripts for running DeepSeek-V4 Flash across 2-node and 3-node topologies.

---

## Recipe Matrix

| Recipe | Base Engine | TP Strategy | Speculative Method | KV Cache Dtype | Orchestration |
| :--- | :--- | :--- | :--- | :--- | :--- |
| [`anemll-v0.25`](./anemll-v0.25/) | vLLM 0.25.1 + Anemll overlay | Attention-group padding (64→72 heads, 8→9 groups) | MTP $K=2$ (deterministic / speculative) | `nvfp4_ds_mla` | Docker Compose / Systemd |
| [`eugr-b12x`](./eugr-b12x/) | B12X runtime engine (`vllm-node-b12x`) | Virtual TP sharding via `--exp-b12x` | DSpark probabilistic ($nst \in [2..7]$) | FP8 / NVFP4 | `eugr-launcher` YAML / Systemd |

---

## Directory Structure

* **[`anemll-v0.25/`](./anemll-v0.25/)**: The baseline runtime lineage (`ghcr.io/anemll/dspark-vllm-gx10:0.1.1` overlay) including the core attention-group padding patch (`patches/apply_tp3_patch.py`), live environment variables (`env/`), and multi-node launch scripts.
* **[`eugr-b12x/`](./eugr-b12x/)**: The B12X runtime configuration recipes (`configs/`), systemd unit files (`systemd/`), and automated benchmark/cell sweep scripts (`scripts/`).

---

## Model Ecosystem References

For Qwen model families evaluated on this cluster, please refer to their dedicated canonical repositories:
* **Qwen 3.8-27B Dense & DFlash2:** [`colonel-otto/3x-dgx-spark-mesh-qwen-3.8-27b-dense`](https://github.com/colonel-otto/3x-dgx-spark-mesh-qwen-3.8-27b-dense)
* **Qwen 3.8 Flash Next (~180B MoE):** [`colonel-otto/3x-dgx-spark-mesh-qwen-3.8-flash-next`](https://github.com/colonel-otto/3x-dgx-spark-mesh-qwen-3.8-flash-next)
