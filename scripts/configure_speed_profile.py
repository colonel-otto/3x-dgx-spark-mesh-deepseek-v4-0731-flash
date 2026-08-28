#!/usr/bin/env python3
import argparse
import pathlib
import re

def update_env_file(path: pathlib.Path, updates: dict):
    if not path.exists():
        return
    text = path.read_text(encoding="utf-8")
    for k, v in updates.items():
        pattern = re.compile(rf"^{k}=.*$", re.MULTILINE)
        if pattern.search(text):
            text = pattern.sub(f"{k}={v}", text)
        else:
            text += f"\n{k}={v}\n"
    path.write_text(text, encoding="utf-8")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", default="/home/sparkmain/localai/dspark-vllm-gx10")
    parser.add_argument("--batched-tokens", type=int, default=16384)
    parser.add_argument("--max-model-len", type=int, default=460800)
    parser.add_argument("--nccl-buffsize", type=int, default=16777216)
    parser.add_argument("--gpu-mem", type=float, default=0.835)
    args = parser.parse_args()

    repo = pathlib.Path(args.repo)
    tp3_env = repo / "config/tp3.env"
    
    updates = {
        "MAX_NUM_BATCHED_TOKENS": str(args.batched_tokens),
        "MAX_MODEL_LEN": str(args.max_model_len),
        "NCCL_BUFFSIZE": str(args.nccl_buffsize),
        "GPU_MEMORY_UTILIZATION": f"{args.gpu_mem:.3f}".rstrip("0").rstrip("."),
        "LONG_PREFILL_TOKEN_THRESHOLD": "1024",
        "DSPARK_MAX_INFLIGHT_PREFILLS": "2",
        "VLLM_PREFIX_CACHE_RETENTION_INTERVAL": "4096",
    }
    update_env_file(tp3_env, updates)
    print(f"Configured speed profile (batched_tokens={args.batched_tokens}, max_len={args.max_model_len}) in {tp3_env}")

if __name__ == "__main__":
    main()
