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
#
# Qwen 3.6 27B was retired on 2026-08-14, superseded by Qwen 3.8 27B, and its weights
# were deleted. Its measurements stay in results.jsonl and in REPORT.md — that study is
# what established the MLX-vs-llama.cpp and warm-cache findings the 3.8 work builds on —
# but it is no longer a benchmark target, and leaving a registry entry pointing at
# missing files would only produce confusing "model not found" runs.
#
# The 35B A3B stays: Qwen 3.8 shipped only 27B and 2.4T-A95B, and the latter does not fit
# in 64 GB, so there is no 3.8 replacement for the fast MoE.
MODELS_CONFIG = {
    "qwen3.6-35b-a3b": {
        "name": "Qwen 3.6 35B A3B",
        "reasoning": True,
        "reasoning_format": "deepseek",
        "quants": {
            "q5": os.path.expanduser("~/Models/qwen3.6-35b-a3b-mtp-q5/Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf"),
            "q6": os.path.expanduser("~/Models/qwen3.6-35b-a3b-mtp-q6/Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf"),
            "q8": os.path.expanduser("~/Models/qwen3.6-35b-a3b-mtp-q8/Qwen3.6-35B-A3B-Q8_0.gguf"),
        },
        "quant_order": ["q5", "q6", "q8"],
        "draft_ns": [0, 1, 2, 3, 4],
    },
    # Qwen 3.8 27B (released 2026-08-14). Dense 64-layer hybrid, 48 GatedDeltaNet + 16
    # full-attention layers — the same 3:1 family as 3.6, and it reports `model_type:
    # qwen3_5`, so llama.cpp and mlx_vlm load it on their existing Qwen3.5 paths.
    #
    # Quant selection is the point of this arm. Qwen themselves publish no GGUF and no
    # MLX (only FP8, which is CUDA-only), so on Apple Silicon there is no official-vs-
    # community choice to make — the real question is which *community* quant is best,
    # and nobody has measured it for 3.8. A KL study on the previous generation found
    # unsloth's UD-Q8_K_XL off the quality/size Pareto frontier while a plain Q8_0 was
    # both smaller and closer to BF16, so the ggml-org quants are carried as a
    # cross-uploader control rather than assuming UD wins by construction.
    #
    # `draft_sidecar`: every unsloth quant carries the MTP head inline (verified by
    # reading the GGUF tensor names: blk.64.nextn.*), so they need no extra file. The
    # ggml-org quants carry none and must be pointed at an `mtp-*.gguf` sidecar.
    "qwen3.8-27b": {
        "name": "Qwen 3.8 27B",
        "reasoning": True,
        "reasoning_format": "deepseek",
        "quants": {
            "q4_ud": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf"),
            "q5": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-Q5_K_M.gguf"),
            "q5_ud": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q5_K_XL.gguf"),
            "q6": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-Q6_K.gguf"),
            "q6_ud": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q6_K_XL.gguf"),
            "q8": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-Q8_0.gguf"),
            "q8_ud": os.path.expanduser("~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q8_K_XL.gguf"),
            # Cross-uploader control (no inline MTP; see draft_sidecars below).
            "q4_ggml": os.path.expanduser("~/Models/qwen3.8-27b-gguf-ggml/Qwen3.8-27B-Q4_K_M.gguf"),
            "q8_ggml": os.path.expanduser("~/Models/qwen3.8-27b-gguf-ggml/Qwen3.8-27B-Q8_0.gguf"),
        },
        # Screening order: the reference quant first, so an interrupted run still has
        # the baseline every other quant is compared against.
        "quant_order": ["q8", "q8_ud", "q8_ggml", "q6_ud", "q6",
                        "q5_ud", "q5", "q4_ud", "q4_ggml"],
        "draft_sidecars": {
            "q4_ggml": os.path.expanduser("~/Models/qwen3.8-27b-gguf-ggml/mtp-Qwen3.8-27B-Q8_0.gguf"),
            "q8_ggml": os.path.expanduser("~/Models/qwen3.8-27b-gguf-ggml/mtp-Qwen3.8-27B-Q8_0.gguf"),
        },
        "draft_ns": [0, 1, 2, 3, 4],
        # Qwen's own thinking-mode recommendation for 3.8 is temp 1.0 (3.6 used 0.6).
        # Applied to both runtimes so the arms stay comparable to each other, while
        # still serving this model the way its authors intend.
        "sampling": {"temp": 1.0, "top_p": 0.95, "top_k": 20},
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
# Served in-process by bench_mlx.py via mlx_vlm, with MTP speculative decoding from the
# separately-published drafter checkpoints.
#
# The drafters are bf16 *on purpose*: quantized MTP heads are reported to collapse
# acceptance on MoE models (single-digit percentages vs ~80% at bf16), because
# quantization error compounds through the expert-routing prediction. They are small
# (~0.8 GB for the 27B head), so keeping them unquantized costs almost nothing.
#
# `draft_block_ns` is the MLX analogue of the GGUF `draft_ns` sweep: 0 means no
# speculation, N overrides the drafter's configured block size (3 for these checkpoints).
#
# Qwen 3.6 27B was retired here on 2026-08-14 along with its GGUF entry; see the note on
# MODELS_CONFIG. Block-size semantics documented there still apply to every drafter: the
# drafter proposes block_size - 1 tokens (speculative/dflash.py:140), so a block size of
# 1 proposes nothing, emits a single token and stops. Valid depths start at 2; 0 is off.
MLX_MODELS_CONFIG = {
    "qwen3.6-35b-a3b": {
        "name": "Qwen 3.6 35B A3B",
        "reasoning": True,
        "quants": {
            "mlx8": os.path.expanduser("~/Models/qwen3.6-35b-a3b-mlx-8bit"),
        },
        "quant_order": ["mlx8"],
        "draft_model": os.path.expanduser("~/Models/qwen3.6-35b-a3b-mtp-bf16"),
        "draft_kind": None,
        # MoE: published acceptance is ~11% (one MTP layer cannot predict expert
        # routing), so this arm confirms or refutes that rather than sweeping.
        # See the 27B note above for why depths start at 2, not 1.
        "draft_block_ns": [0, 2, 3],
        "kv_bits_opts": [None, 8],
        "tiers": ["shallow", "agent"],
    },
    # Qwen 3.8 27B MLX arm. Three targets from two uploaders, because MLX quant quality
    # across sources is exactly as unmeasured as the GGUF side.
    #
    # The drafter is a caveat, not a footnote: mlx-community published no MTP head for
    # 3.8 (they did for 3.6), so the only option is vvsotnikov's, and a drafter is only
    # valid against the checkpoint it was distilled from. Acceptance is therefore
    # expected to hold on the matching 8-bit target and may collapse on the other two.
    # Phase 0 measures that rather than assuming it, and any target where acceptance
    # collapses is benchmarked MTP-off and labelled, not quietly reported as slow.
    "qwen3.8-27b": {
        "name": "Qwen 3.8 27B",
        "reasoning": True,
        "quants": {
            "mlx8": os.path.expanduser("~/Models/qwen3.8-27b-mlx-8bit"),
            "mlx6": os.path.expanduser("~/Models/qwen3.8-27b-mlx-6bit"),
            "mxfp8": os.path.expanduser("~/Models/qwen3.8-27b-mlx-mxfp8"),
        },
        "quant_order": ["mlx8", "mlx6", "mxfp8"],
        "draft_model": os.path.expanduser("~/Models/qwen3.8-27b-mtp-mlx-8bit"),
        # bf16 alternate, kept as the acceptance control: on 3.6 the quantized MTP heads
        # were reported to collapse acceptance, and this drafter is only available
        # quantized from a third party, so the two are compared directly in Phase C.
        "draft_model_alt": os.path.expanduser("~/Models/qwen3.8-27b-mtp-mlx-bf16"),
        "draft_kind": None,           # auto-detected from the drafter's model_type
        "draft_block_ns": [0, 2, 3, 4, 5],
        "kv_bits_opts": [None, 8],
        "tiers": ["shallow", "agent", "deep"],
        "sampling": {"temp": 1.0, "top_p": 0.95, "top_k": 20},
        # Qwen 3.8's template adds a reasoning_effort knob (default "xhigh"). Pinned
        # explicitly so the MLX prompt matches what llama-server's --jinja renders from
        # the same template default; validate_parity.py checks the rendered prompts.
        "reasoning_effort": "xhigh",
    },
}


def sampling_for(model_cfg: dict) -> tuple:
    """(temp, top_p, top_k) for a model — its own recommendation, else the global default.

    Qwen 3.8 asks for temp 1.0 in thinking mode where 3.6 asked for 0.6. Hardcoding one
    value across models would mean serving at least one of them wrong, so it is per-model
    but shared across runtimes: the GGUF-vs-MLX comparison stays like-for-like, which is
    the invariant that actually matters here.
    """
    s = (model_cfg or {}).get("sampling") or {}
    return s.get("temp", TEMP), s.get("top_p", TOP_P), s.get("top_k", TOP_K)

# --- MLX generation knobs -------------------------------------------------------
KV_GROUP_SIZE = 64
KV_QUANT_SCHEME = "uniform"   # "turboquant" also fails with MTP; see note below
PREFILL_STEP_SIZE = 2048      # do NOT raise: prefilling 23k in one step OOMs Metal

# Quantized KV + MTP is broken in mlx_vlm 0.6.3 at depth.
#
#   mlx_vlm/models/qwen3_5/language.py:1481
#     prefix_len = keys.shape[-2] - L      # `keys` is a list when the cache is quantized
#
# A quantized MLX KV cache is a list of (values, scales, biases); the speculative-verify
# branch of Qwen3.5 attention indexes it as a plain array. Measured matrix on this box:
# shallow+q8+MTP passes, agent+fp16+MTP passes, agent+q8+no-MTP passes, agent+q8+MTP
# fails for every block size. `turboquant` fails differently
# ("'_QuantizedStateProxy' object is not subscriptable").
#
# Deferring quantization past the prompt avoids the crash, because the prompt's KV then
# stays unquantized and only decode-time KV is quantized. That is NOT equivalent to
# llama.cpp's -ctk q8_0 -ctv q8_0, which quantizes the whole cache — it forfeits exactly
# the memory saving that makes q8 KV worth using at long context. It is measured here as
# its own labelled arm ("q8 decode-only") rather than being passed off as q8 KV.
QUANTIZED_KV_DEFER = True


def quantized_kv_start_for(tier: str) -> int:
    """Step at which to begin quantizing the KV cache, per prompt-depth tier.

    Set just past the tier's prompt so prefill never hands a quantized cache to the
    speculative verify path. Returns 0 when deferral is off (which will crash with MTP
    at depth — kept switchable so the bug can be re-tested against future mlx_vlm).
    """
    if not QUANTIZED_KV_DEFER:
        return 0
    return {"shallow": 1024, "agent": 24000, "deep": 66000}.get(tier, 1024)


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


def ctx_for_tier(tier: str, n_predict: int) -> int:
    """Context large enough to hold the tier's prompt plus the generation cap.

    The old harness pinned -c 16384 for everything. That cannot hold the `agent`
    (~23k) or `deep` (~64k) prompts at all, and it also means the GGUF control was
    measured at a different context size than the MLX arm. Both runtimes now size
    context from the same tier.
    """
    prompt_tokens = {"shallow": 512, "agent": 23000, "deep": 64000}.get(tier, 512)
    need = prompt_tokens + n_predict + 2048   # headroom for template + slack
    # Round up to the next power-of-two-ish boundary llama.cpp is happy with.
    for size in (16384, 32768, 65536, 98304, 131072, 262144):
        if need <= size:
            return size
    return 262144


def server_cmd(model_path: str, draft_n: int, log_path: str, reasoning_format: str = "none",
               ctx: int = None, kv_quant: str = None, draft_sidecar: str = None) -> list:
    """Build the llama-server argv for one (quant, draft_n) configuration.

    `kv_quant` (e.g. "q8_0") mirrors the MLX arm's --kv-bits so neither runtime gets a
    free ride on KV cache precision — and it matches what llm-serve actually serves with
    (LOCAL_LLM_HARNESS.md §4 uses -ctk q8_0 -ctv q8_0).

    `draft_sidecar` is for quants that ship the MTP head as a separate file instead of
    inline. The unsloth Qwen 3.8 quants embed it (blk.64.nextn.*) and need nothing; the
    ggml-org ones do not and must be handed an `mtp-*.gguf` here, or they would silently
    benchmark as if speculation were unavailable.
    """
    cmd = [
        LLAMA_SERVER,
        "-m", model_path,
        "-c", str(ctx or CTX),
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
    if kv_quant:
        cmd += ["-ctk", kv_quant, "-ctv", kv_quant]
    if reasoning_format and reasoning_format != "none":
        cmd += ["--reasoning-format", reasoning_format]

    if draft_n == 0:
        cmd += ["--spec-type", "none"]
    else:
        cmd += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n)]
        if draft_sidecar:
            cmd += ["--model-draft", draft_sidecar]
    return cmd
