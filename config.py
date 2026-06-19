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
# Capped at 4: the sweep showed acceptance and tok/s both fall off past n=4, so n=5-8
# are excluded from analysis/recommendation (raw records remain in results.jsonl).
DRAFT_NS = [0, 1, 2, 3, 4]

# Single pass. (Outputs are deterministic across draft-n with a fixed seed, so the
# speed sweep needs only one measurement per cell.)
PASSES = [1]

# --- Decoding (fixed across every run) -----------------------------------------
SEED = 42
TEMP = 0.6
TOP_P = 0.95
TOP_K = 20
CTX = 16384               # safely holds prompt (~1-2k) + up to 8192 generated

# Two-phase design (keeps the run to ~5-6h while preserving every metric):
#   * phase "speed": short cap on the FULL grid (quants x draft_n x questions) -- tok/s,
#     prompt speed, TTFT, MTP acceptance. tok/s is cap-independent, so a short cap is
#     sufficient and fair (all configs generate the same deterministic prefix).
#   * phase "full":  the real 8192 cap, but only off-config x quant x question (15 runs)
#     -- complete answers for judging + true thinking/answer token totals. Draft-n does
#     not change the output, so one config suffices.
SPEED_N_PREDICT = 1024
# Total-token cap (thinking + answer share one budget). 12288 lets the answer complete
# even on the long coding prompt (thinking self-terminates ~5-7k for these prompts, so
# this leaves ample room for the answer) while fitting in CTX=16384 with the prompt.
FULL_N_PREDICT = 12288

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
