#!/usr/bin/env python3
"""Fetch the MLX 8-bit targets and their bf16 MTP drafters.

Order matters: the 27B drafter and target come first so the MTP smoke gate can run
while the (much larger) 35B pair is still downloading.

Run under the venv:  .mlxenv/bin/python download_mlx_models.py
"""
import os
import sys
import time

from huggingface_hub import snapshot_download

MODELS_DIR = os.path.expanduser("~/Models")

# (repo_id, local_dir_name) in smoke-gate-first order.
TARGETS = [
    # The 27B pair was removed when Qwen 3.6 27B was retired for 3.8; see
    # download_qwen38.py for the current 27B-class targets.
    ("mlx-community/Qwen3.6-35B-A3B-MTP-bf16", "qwen3.6-35b-a3b-mtp-bf16"),
    ("lmstudio-community/Qwen3.6-35B-A3B-MLX-8bit", "qwen3.6-35b-a3b-mlx-8bit"),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def main():
    os.makedirs(MODELS_DIR, exist_ok=True)
    for repo_id, name in TARGETS:
        dest = os.path.join(MODELS_DIR, name)
        if os.path.isfile(os.path.join(dest, "config.json")):
            log(f"skip {name} (already present)")
            continue
        log(f"fetching {repo_id} -> {dest}")
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest,
                # Weights + config + tokenizer only; skip duplicate .bin/.pth and docs.
                ignore_patterns=["*.pt", "*.bin", "*.pth", "*.md", "*.gguf", "original/*"],
                max_workers=8,
            )
        except Exception as e:
            log(f"FAILED {repo_id}: {type(e).__name__}: {e}")
            continue
        log(f"done {name} in {(time.time()-t0)/60:.1f} min")
    log("ALL DOWNLOADS FINISHED")


if __name__ == "__main__":
    sys.exit(main())
