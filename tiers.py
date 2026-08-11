"""Prompt-depth tiers.

The old benchmark measured everything at ~200-token prompts. The actual workload —
Claude Code driving a local model — sends ~23k tokens before the user even types
(LOCAL_LLM_HARNESS.md §4), and prompt-processing throughput decays sharply with depth.
A runtime comparison made at 200 tokens does not predict behaviour at 23k.

So each question is measured at three depths. Filler is prepended to the user turn as a
CONTEXT block, mirroring how an agent harness ships repo context ahead of the real ask.

Two properties matter and are enforced here:

  * **Identical text across runtimes.** The built prompts are cached to
    results/prompt_tiers.json and read by both bench.py (llama.cpp) and bench_mlx.py.
    Both runtimes tokenize the same bytes, so prompt lengths match by construction
    rather than by luck.
  * **Realistic filler.** Filler is drawn from real source files in the user's repos,
    in deterministic sorted order. Synthetic or repeated boilerplate would be far more
    predictable than real context and would inflate MTP acceptance — the exact metric
    under test.
"""
import json
import os

import config as C

# Target *total* prompt tokens per tier. "shallow" means no filler at all.
TIERS = {
    "shallow": None,      # ~200 tok — comparable to the existing REPORT.md numbers
    "agent": 23000,       # real Claude Code preamble depth
    "deep": 64000,        # deep agent session
}
TIER_ORDER = ["shallow", "agent", "deep"]

CACHE_PATH = os.path.join(C.RESULTS, "prompt_tiers.json")

# Real files, deterministic order. Read as text; anything missing is skipped.
_CORPUS_ROOTS = [
    os.path.join(C.REPO, "bench.py"),
    os.path.join(C.REPO, "report.py"),
    os.path.join(C.REPO, "config.py"),
    os.path.join(C.REPO, "questions.py"),
    os.path.join(C.REPO, "RESEARCH_models.md"),
    os.path.expanduser("~/Desktop/Programming/local-setup/LOCAL_LLM_HARNESS.md"),
    os.path.expanduser("~/Desktop/Programming/local-setup/README.md"),
    os.path.expanduser("~/Desktop/Programming/local-setup/scripts/llm-proxy.mjs"),
    os.path.expanduser("~/Desktop/Programming/local-setup/scripts/llm-serve"),
    os.path.expanduser("~/Desktop/Programming/local-setup/setup.sh"),
]


def _corpus() -> str:
    """Concatenate real repo files, deterministically, with file headers."""
    parts = []
    for path in _CORPUS_ROOTS:
        try:
            with open(path, "r", errors="replace") as f:
                body = f.read()
        except OSError:
            continue
        parts.append(f"===== FILE: {path} =====\n{body}\n")
    if not parts:
        raise RuntimeError("tiers.py: no corpus files readable; cannot build depth tiers")
    return "\n".join(parts)


def _fit_to_tokens(text: str, target: int, count_tokens) -> str:
    """Trim/extend `text` so it measures ~`target` tokens under `count_tokens`.

    Binary search on character length. Exactness is not required — what matters is
    that both runtimes get the *same string*, and that it lands near the target.
    """
    if count_tokens(text) < target:
        reps = (target // max(1, count_tokens(text))) + 2
        text = "\n\n".join([text] * reps)

    lo, hi = 0, len(text)
    best = text
    while lo < hi:
        mid = (lo + hi) // 2
        n = count_tokens(text[:mid])
        if n < target:
            lo = mid + 1
        else:
            best = text[:mid]
            hi = mid
    return best


def build(questions, count_tokens, force: bool = False) -> dict:
    """Build (or load) the tiered prompts.

    `count_tokens(str) -> int` should come from the model's own tokenizer. The result is
    cached so that the second runtime reads byte-identical prompts rather than rebuilding
    them against a different tokenizer.

    Returns {question_id: {tier: user_text}}.
    """
    if not force and os.path.exists(CACHE_PATH):
        with open(CACHE_PATH) as f:
            return json.load(f)

    corpus = _corpus()
    out = {}
    for q in questions:
        out[q["id"]] = {}
        for tier, target in TIERS.items():
            if target is None:
                out[q["id"]][tier] = q["user"]
                continue
            # Reserve room for the question itself so the total lands near target.
            budget = target - count_tokens(q["user"]) - 64
            filler = _fit_to_tokens(corpus, max(1, budget), count_tokens)
            out[q["id"]][tier] = (
                "Here is the working context for this session.\n\n"
                "<context>\n" + filler + "\n</context>\n\n"
                "Now answer the following.\n\n" + q["user"]
            )

    os.makedirs(C.RESULTS, exist_ok=True)
    with open(CACHE_PATH, "w") as f:
        json.dump(out, f)
    return out


def load_cached() -> dict:
    """Read the prompts built by the MLX side.

    The GGUF harness must send byte-identical text, so it reads the cache rather than
    rebuilding against llama.cpp's tokenizer — rebuilding would risk two runtimes being
    compared on prompts that differ by a few tokens.
    """
    if not os.path.exists(CACHE_PATH):
        raise RuntimeError(
            f"{CACHE_PATH} missing. Build it first with the MLX tokenizer:\n"
            f"  .mlxenv/bin/python -c 'import tiers,questions;"
            f"from transformers import AutoTokenizer;"
            f"t=AutoTokenizer.from_pretrained(\"~/Models/qwen3.6-27b-mtp-bf16\");"
            f"tiers.build(questions.QUESTIONS, lambda s: len(t.encode(s)))'"
        )
    with open(CACHE_PATH) as f:
        return json.load(f)


def user_text(built: dict, question_id: str, tier: str) -> str:
    return built[question_id][tier]
