"""Central configuration for the local LLM benchmark.

Everything tunable lives here so bench.py / report.py / judging stay declarative.
"""
import os

REPO = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(REPO, "results")
LOGS = os.path.join(RESULTS, "logs")
CHARTS = os.path.join(RESULTS, "charts")
JUDGING = os.path.join(RESULTS, "judging")

RESULTS_JSONL = os.path.join(RESULTS, "results.jsonl")
RESULTS_JSON = os.path.join(RESULTS, "results.json")
SUMMARY_JSON = os.path.join(RESULTS, "summary.json")
PROGRESS_LOG = os.path.join(RESULTS, "progress.log")

# --- Model Registry -----------------------------------------------------------
# Registered models, their characteristics, paths, and sweep parameters.
MODELS_CONFIG = {
    "qwen3.6-27b": {
        "name": "Qwen 3.6 27B",
        "reasoning": True,
        "reasoning_format": "deepseek",  # llama-server option
        "quants": {
            "q5": os.path.expanduser("~/Models/qwen3.6-27b-mtp-q5/Qwen3.6-27B-UD-Q5_K_XL.gguf"),
            "q6": os.path.expanduser("~/Models/qwen3.6-27b-mtp-q6/Qwen3.6-27B-UD-Q6_K_XL.gguf"),
            "q8": os.path.expanduser("~/Models/qwen3.6-27b-mtp-q8/Qwen3.6-27B-Q8_0.gguf"),
        },
        "quant_order": ["q5", "q6", "q8"],
        "draft_ns": [0, 1, 2, 3, 4],
    },
    "gemma4-31b": {
        "name": "Gemma 4 31B",
        "reasoning": False,
        "reasoning_format": "none",
        "quants": {
            "q5": os.path.expanduser("~/Models/gemma4-31b/Gemma-4-31B-Q5_K_M.gguf"),
            "q8": os.path.expanduser("~/Models/gemma4-31b/Gemma-4-31B-Q8_0.gguf"),
        },
        "quant_order": ["q5", "q8"],
        "draft_ns": [0],  # Standard GGUF with no speculative MTP configuration
    },
}

# --- MLX Model Registry --------------------------------------------------------
MLX_MODELS_CONFIG = {
    "qwen3.6-27b": {
        "name": "Qwen 3.6 27B",
        "reasoning": True,
        "quants": {
            "mlx8": os.path.expanduser("~/Models/qwen3.6-27b-mlx-8bit"),
            "mlx6": os.path.expanduser("~/Models/qwen3.6-27b-mlx-6bit"),
        },
        "quant_order": ["mlx8", "mlx6"],
    },
    "gemma4-31b": {
        "name": "Gemma 4 31B",
        "reasoning": False,
        "quants": {
            "mlx8": os.path.expanduser("~/Models/gemma4-31b-mlx-8bit"),
        },
        "quant_order": ["mlx8"],
    },
}


# --- Decoding (fixed across every run) -----------------------------------------
SEED = 42
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
CTX = 16384               # safely holds prompt (~1-2k) + up to 8192 generated

# Two-phase design (keeps the run to ~5-6h while preserving every metric):
SPEED_N_PREDICT = 1024
FULL_N_PREDICT = 12288

# Smoke test overrides (fast end-to-end pipeline validation)
SMOKE_N_PREDICT = 512
SMOKE_PASSES = [1]

# --- Server -------------------------------------------------------------------
LLAMA_SERVER = "/opt/homebrew/bin/llama-server"
HOST = "127.0.0.1"
PORT = 8089
BASE_URL = f"http://{HOST}:{PORT}"

HEALTH_TIMEOUT_S = 300    # max wait for a model to load + warm up
REQ_TIMEOUT_S = 1800      # max wall time for a single generation
STREAM_READ_TIMEOUT_S = 240  # max gap between streamed tokens before we bail
PORT_FREE_TIMEOUT_S = 60  # max wait for the port to free after server stop


def server_cmd(model_path: str, draft_n: int, log_path: str, reasoning_format: str = "none") -> list:
    """Build the llama-server argv for one (quant, draft_n) configuration."""
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "-c", str(CTX),
        "-ngl", "99",
        "-fa", "on",
        "-np", "1",                 # MTP requires a single slot
        "--jinja",
        "-s", str(SEED),
        "--host", HOST,
        "--port", str(PORT),
        "--metrics",
        "--no-webui",
    ]
    if reasoning_format and reasoning_format != "none":
        cmd += ["--reasoning-format", reasoning_format]

    if draft_n == 0:
        cmd += ["--spec-type", "none"]
    else:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n)]
    return cmd
