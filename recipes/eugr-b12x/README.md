# Recipe: Eugr B12X Virtual TP Runtime (`eugr-b12x`)

This recipe packages the B12X virtual tensor parallel sharding engine for DeepSeek-V4 Flash across the 3x DGX Spark mesh cluster.

---

## Overview & Architecture

* **Container Engine:** `vllm-node-b12x` (pulled by digest / container runtime)
* **TP Strategy:** Virtual TP sharding via `--exp-b12x` flag (handles 64-head padding internally).
* **Speculative Decoding:** DSpark probabilistic speculative decoding ($nst \in [2..7]$, $mnbt \in [8192, 16384]$).
* **KV Cache Dtype:** FP8 / NVFP4.
* **Orchestration:** Managed via `eugr-launcher` YAML recipes and systemd unit service (`eugr.service`).

---

## Directory Contents

* **`configs/`**: Nine YAML recipe configurations (seven operational configs plus two base templates):
  * **Operational Configurations (7):**
    * `dsv4-flash-0731-local-tp3.yaml`: Baseline 3-node TP=3 config ($nst=5$, $mnbt=8192$).
    * `dsv4-flash-0731-local-tp2.yaml`: 2-node TP=2 comparison config ($nst=5$, $mnbt=8192$).
    * `dsv4-tp3-nst2-mnbt8192.yaml`: Speculative token sweep ($nst=2$, $mnbt=8192$).
    * `dsv4-tp3-nst3-mnbt8192.yaml`: Speculative token sweep ($nst=3$, $mnbt=8192$).
    * `dsv4-tp3-nst5-mnbt8192.yaml`: Speculative token sweep ($nst=5$, $mnbt=8192$).
    * `dsv4-tp3-nst7-mnbt8192.yaml`: Speculative token sweep ($nst=7$, $mnbt=8192$).
    * `dsv4-tp3-nst5-mnbt16384.yaml`: High batch token limit config ($nst=5$, $mnbt=16384$).
  * **Base Templates (2):**
    * `deepseek-v4-flash-0731.yaml`: Upstream base template for 0731 snapshot.
    * `deepseek-v4-flash.yaml`: Upstream generic base template.
* **`systemd/`**: Systemd unit files and service control scripts:
  * `eugr.service`: Unit definition for systemd management.
  * `eugr-service-start` / `eugr-service-stop`: Service lifecycle scripts.
  * `install-service.sh`: Installer script for host nodes.
* **`scripts/`**: Automation, validation, and benchmarking utilities:
  * `eugr-boot.sh`: Cluster bootloader for B12X recipes.
  * `eugr-sweep.sh`: Automated multi-configuration runner.
  * `quick-validate.sh` & `verify-gateway.sh`: Healthcheck and API endpoint verifiers.
  * `eugr-remaining-cells.py` / `eugr-remaining-cells-v2.py`: Grid completion trackers.
  * `bench-miaai.py`: Matched benchmark harness.

---

## Quick Start

### 1. Launch via Systemd / Boot Script
```bash
# Boot the 3-node TP=3 recipe
bash scripts/eugr-boot.sh configs/dsv4-flash-0731-local-tp3.yaml
```

### 2. Validate Endpoint
```bash
bash scripts/quick-validate.sh
```

### 3. Run Benchmark Suite
```bash
bash scripts/eugr-sweep.sh
```
