#!/usr/bin/env python3
"""Fetch the Qwen 3.8 27B benchmark candidates.

Ordering is gate-first, not size-first: the MLX drafter (451 MB) and its matching 8-bit
target come before anything else so the MTP smoke gate can run while the ~170 GB of GGUFs
are still streaming. A failed gate should cost minutes, not hours.

Two facts drive the candidate list, both verified against the GGUF headers on 2026-08-14
by range-fetching the first 20 MB of each file and grepping the tensor names:

  * Every unsloth quant carries the MTP head inline (`blk.64.nextn.{eh_proj,enorm,hnorm,
    shared_head_norm}`, arch `qwen35`), so llama.cpp gets `--spec-type draft-mtp` with no
    extra file. Unlike Qwen 3.6 there is no separate `-MTP-GGUF` repo, and none is needed.
  * The ggml-org quants carry none — that repo ships MTP as `mtp-Qwen3.8-27B-*.gguf`
    sidecars, which have to be passed with `--model-draft`. Hence the sidecar download.

The ggml-org quants are the cross-uploader control. A KL study on the previous generation
found unsloth's UD-Q8_K_XL *off* the quality/size Pareto frontier while a plain Q8_0 was
both smaller and closer to BF16, so "UD is strictly better" is an assumption worth
measuring rather than inheriting.

    .mlxenv/bin/python download_qwen38.py            # everything, in gate order
    .mlxenv/bin/python download_qwen38.py --gate     # only what Phase 0 needs
"""
import argparse
import os
import sys
import time

from huggingface_hub import snapshot_download

MODELS_DIR = os.path.expanduser("~/Models")

# (repo_id, local_dir_name, allow_patterns or None, is_gate)
TARGETS = [
    # --- Phase 0 gate assets ---------------------------------------------------------
    ("vvsotnikov/Qwen3.8-27B-MTP-MLX-8bit", "qwen3.8-27b-mtp-mlx-8bit", None, True),
    ("mlx-community/Qwen3.8-27B-8bit", "qwen3.8-27b-mlx-8bit", None, True),
    # Mid-size quant, big enough to be representative and small enough to gate on.
    ("unsloth/Qwen3.8-27B-GGUF", "qwen3.8-27b-gguf",
     ["Qwen3.8-27B-UD-Q6_K_XL.gguf"], True),

    # --- MLX arm ---------------------------------------------------------------------
    # bf16 drafter as an acceptance control: on Qwen 3.6 the quantized MTP heads were
    # reported to collapse acceptance, so measure rather than assume it is safe here.
    ("vvsotnikov/Qwen3.8-27B-MTP-MLX-bf16", "qwen3.8-27b-mtp-mlx-bf16", None, False),
    ("mlx-community/Qwen3.8-27B-mxfp8", "qwen3.8-27b-mlx-mxfp8", None, False),
    ("lmstudio-community/Qwen3.8-27B-MLX-6bit", "qwen3.8-27b-mlx-6bit", None, False),

    # --- GGUF arm: the contenders that could plausibly be a daily driver --------------
    ("unsloth/Qwen3.8-27B-GGUF", "qwen3.8-27b-gguf", [
        "Qwen3.8-27B-Q8_0.gguf",
        "Qwen3.8-27B-UD-Q8_K_XL.gguf",
        "Qwen3.8-27B-Q6_K.gguf",
        "Qwen3.8-27B-UD-Q5_K_XL.gguf",
        "Qwen3.8-27B-Q5_K_M.gguf",
        "Qwen3.8-27B-UD-Q4_K_XL.gguf",
    ], False),

    # --- GGUF cross-uploader control + its MTP sidecars -------------------------------
    ("ggml-org/Qwen3.8-27B-GGUF", "qwen3.8-27b-gguf-ggml", [
        "Qwen3.8-27B-Q8_0.gguf",
        "Qwen3.8-27B-Q4_K_M.gguf",
        "mtp-Qwen3.8-27B-Q8_0.gguf",
        "mtp-Qwen3.8-27B-BF16.gguf",
    ], False),
]


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def _present(dest, allow_patterns):
    """Already-downloaded check. Per-file for GGUF repos, config.json for MLX repos."""
    if allow_patterns:
        return all(os.path.isfile(os.path.join(dest, p)) for p in allow_patterns)
    return os.path.isfile(os.path.join(dest, "config.json"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gate", action="store_true",
                    help="fetch only the assets Phase 0 needs, then stop")
    args = ap.parse_args()

    os.makedirs(MODELS_DIR, exist_ok=True)
    failures = []
    for repo_id, name, allow, is_gate in TARGETS:
        if args.gate and not is_gate:
            continue
        dest = os.path.join(MODELS_DIR, name)
        if _present(dest, allow):
            log(f"skip {name} ({repo_id.split('/')[0]}, already present)")
            continue
        log(f"fetching {repo_id} -> {name}")
        t0 = time.time()
        try:
            snapshot_download(
                repo_id=repo_id,
                local_dir=dest,
                allow_patterns=allow,
                # Weights + config + tokenizer only; skip torch duplicates and docs.
                ignore_patterns=None if allow else
                ["*.pt", "*.bin", "*.pth", "*.md", "*.gguf", "original/*"],
                max_workers=8,
            )
        except Exception as e:
            log(f"FAILED {repo_id}: {type(e).__name__}: {e}")
            failures.append(repo_id)
            continue
        log(f"done {name} in {(time.time() - t0) / 60:.1f} min")

    log("GATE DOWNLOADS FINISHED" if args.gate else "ALL DOWNLOADS FINISHED")
    if failures:
        log(f"FAILURES: {failures}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
