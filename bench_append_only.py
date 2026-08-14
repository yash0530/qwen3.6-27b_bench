#!/usr/bin/env python3
"""Can MLX reuse a prompt prefix for Qwen 3.6 if the prompt is built append-only?

Background. mlx_vlm's Automatic Prompt Caching never engages for Qwen 3.6: measured
lookups_hit=0 / stores=0 with APC_ENABLED=1, and TTFT flat across a 5-turn session. The
cause is architectural — Qwen 3.6 is hybrid (qwen3_5/language.py:2615 returns ArraysCache
for GatedDeltaNet layers and KVCache for attention layers), and prefix reuse requires
rewinding the cache to the common-prefix boundary. ArraysCache has no trim() and
is_trimmable() returns False, so can_trim_prompt_cache() is False. See
ml-explore/mlx-lm#980.

But the rewind is only needed because apply_chat_template re-renders history and drops the
<think> block, so turn N+1's prompt is not a strict extension of what the cache holds. The
cache after turn N holds exactly (prompt_ids + generated_ids). Build turn N+1 by APPENDING
to that token sequence and it strictly extends the cache: no trim, no rewind, and the
recurrent state simply evolves forward — the direction the research says is safe.

This measures that path, and checks it is output-preserving by generating the same
continuation cold (fresh cache) and warm (live cache) at temperature 0.

    .mlxenv/bin/python bench_append_only.py --model qwen3.6-35b-a3b

Result on this machine: 27B 55.39s -> 0.48s (115x), 35B 12.51s -> 0.20s (64x, and
byte-identical to cold). Implication: MLX's warm-cache deficit is a serving-layer gap,
not an engine limitation.
"""
import argparse
import hashlib
import json
import os
import time
from datetime import datetime, timezone

import mlx.core as mx
from mlx_lm.models.cache import make_prompt_cache
from mlx_vlm import apply_chat_template, load, stream_generate

import config as C
import tiers as T
from questions import QUESTIONS

OUT = os.path.join(C.RESULTS, "append_only.jsonl")
FOLLOW_UP = "Name the single biggest risk in one sentence."


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-27b", choices=list(C.MLX_MODELS_CONFIG))
    ap.add_argument("--tier", default="agent", choices=T.TIER_ORDER)
    ap.add_argument("--max-tokens", type=int, default=40)
    args = ap.parse_args()

    cfg = C.MLX_MODELS_CONFIG[args.model]
    path = cfg["quants"]["mlx8"]
    model, processor = load(path)
    tok = getattr(processor, "tokenizer", processor)
    lm = model.language_model if hasattr(model, "language_model") else model
    built = T.load_cached()
    q = QUESTIONS[0]

    def gen(ids, cache):
        t0 = time.time(); first = None; out = []; toks = []
        for r in stream_generate(model, processor, "", input_ids=mx.array([ids]),
                                 prompt_cache=cache, max_tokens=args.max_tokens,
                                 temperature=0.0, seed=C.SEED,
                                 prefill_step_size=C.PREFILL_STEP_SIZE):
            if r.text:
                first = first or time.time()
                out.append(r.text)
            if getattr(r, "token", None) is not None:
                toks.append(r.token)
        return (first - t0 if first else None), "".join(out), toks

    msgs = [{"role": "system", "content": q["system"]},
            {"role": "user", "content": T.user_text(built, q["id"], args.tier)}]
    ids1 = tok.encode(apply_chat_template(processor, model.config, msgs,
                                          add_generation_prompt=True, enable_thinking=True))

    cache = make_prompt_cache(lm)
    kinds = sorted({type(c).__name__ for c in cache})
    t1, _, gen1 = gen(ids1, cache)
    print(f"cache entry types: {kinds}")
    print(f"turn 1 (cold, {len(ids1)} tok): TTFT {t1:.2f}s, generated {len(gen1)} tok")

    # Append to the exact sequence the cache holds, rather than re-rendering the template.
    suffix = tok.encode(f"<|im_end|>\n<|im_start|>user\n{FOLLOW_UP}<|im_end|>\n"
                        f"<|im_start|>assistant\n<think>\n", add_special_tokens=False)
    full2 = ids1 + gen1 + suffix

    t_warm, warm_text, _ = gen(suffix, cache)
    print(f"turn 2 WARM (continuation, {len(suffix)} new tok): TTFT {t_warm:.2f}s")

    t_cold, cold_text, _ = gen(full2, make_prompt_cache(lm))
    print(f"turn 2 COLD ({len(full2)} tok, fresh cache): TTFT {t_cold:.2f}s")

    same = warm_text.strip() == cold_text.strip()
    speedup = t_cold / t_warm if t_warm else None
    print(f"\nSPEED  : {t_cold:.2f}s -> {t_warm:.2f}s ({speedup:.0f}x)")
    print(f"CORRECT: warm output {'MATCHES' if same else 'DIVERGES FROM'} cold")

    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "model": args.model, "tier": args.tier, "cache_kinds": kinds,
        "prompt_tokens": len(ids1), "generated_turn1": len(gen1),
        "delta_tokens": len(suffix), "full_turn2_tokens": len(full2),
        "ttft_cold_turn1_s": t1, "ttft_warm_s": t_warm, "ttft_cold_s": t_cold,
        "speedup": speedup, "output_preserved": same,
        "warm_sha256": hashlib.sha256(warm_text.encode()).hexdigest(),
        "cold_sha256": hashlib.sha256(cold_text.encode()).hexdigest(),
    }
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    print(f"wrote -> {OUT}")


if __name__ == "__main__":
    main()
