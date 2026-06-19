"""Central configuration for the Qwen3.6-27B MTP benchmark.

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

# --- Models under test (the three downloaded quants) ---------------------------
MODELS = {
    "q5": os.path.expanduser("~/Models/qwen3.6-27b-mtp-q5/Qwen3.6-27B-UD-Q5_K_XL.gguf"),
    "q6": os.path.expanduser("~/Models/qwen3.6-27b-mtp-q6/Qwen3.6-27B-UD-Q6_K_XL.gguf"),
    "q8": os.path.expanduser("~/Models/qwen3.6-27b-mtp-q8/Qwen3.6-27B-Q8_0.gguf"),
}
# Order quants are exercised in (smallest first = fastest to load while smoke-debugging).
QUANT_ORDER = ["q5", "q6", "q8"]

# --- MTP sweep -----------------------------------------------------------------
# draft_n == 0 is the special "MTP off" baseline (--spec-type none).
DRAFT_NS = [0, 1, 2, 3, 4, 5, 6, 7, 8]

# Two full passes: with a fixed seed the *outputs* are identical across passes,
# so the 2nd pass exists purely to average timing noise (tok/s, TTFT) and report spread.
PASSES = [1, 2]

# --- Decoding (fixed across every run) -----------------------------------------
SEED = 42
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
N_PREDICT = 8192          # generation cap per answer
CTX = 16384               # safely holds prompt (~1-2k) + 8192 generated

# Smoke test overrides (fast end-to-end pipeline validation)
SMOKE_QUANTS = ["q5"]
SMOKE_DRAFT_NS = [0, 3]
SMOKE_QUESTION_IDS = ["q2_coding"]
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


def server_cmd(model_path: str, draft_n: int, log_path: str) -> list:
    """Build the llama-server argv for one (quant, draft_n) configuration."""
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "-c", str(CTX),
        "-ngl", "99",
        "-fa", "on",
        "-np", "1",                 # MTP requires a single slot
        "--jinja",
        "--reasoning-format", "deepseek",
        "-s", str(SEED),
        "--host", HOST,
        "--port", str(PORT),
        "--metrics",
        "--no-webui",
    ]
    if draft_n == 0:
        cmd += ["--spec-type", "none"]
    else:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n)]
    return cmd
