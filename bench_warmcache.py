#!/usr/bin/env python3
"""Warm-cache TTFT: what an agent loop actually pays after the first turn.

The main sweep measures cold prefill — GGUF with `cache_prompt: false`, MLX verified at
`cached_tokens == 0`. That is the right control for comparing engines, but it is the worst
case, and it is not what a Claude Code session experiences. There the ~23k preamble is
stable across turns and both engines cache it (llama.cpp slot cache; mlx_vlm's APC, with
/v1/cache/stats). If a warm cache collapses TTFT, the 25.6s prefill advantage MLX shows at
the 27B largely evaporates and the decision swings back toward raw decode.

This simulates a real multi-turn session: a fixed system preamble plus a conversation that
grows by one exchange per turn. Turn 1 is cold. Turns 2..N should hit the cached prefix and
only prefill the delta. What matters is the TTFT curve across turns, per engine.

Usage:
  .mlxenv/bin/python bench_warmcache.py --arm mlx      --model qwen3.8-27b
  python3            bench_warmcache.py --arm gguf     --model qwen3.8-27b
"""
import argparse
import json
import os
import time
import urllib.request
from datetime import datetime, timezone

import config as C
import tiers as T
from questions import QUESTIONS
from bench_concurrency import Server, server_argv, BASE, PORT, log

RESULTS_JSONL = os.path.join(C.RESULTS, "warmcache.jsonl")
N_TURNS = 5
GEN_TOKENS = 96          # short: this measures TTFT, not decode


def one_turn(messages, model_name, n_predict, cache_prompt):
    payload = {"messages": messages, "max_tokens": n_predict, "stream": True,
               "model": model_name, "temperature": C.TEMP, "top_p": C.TOP_P}
    if cache_prompt is not None:
        payload["cache_prompt"] = cache_prompt
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; out = []
    with urllib.request.urlopen(req, timeout=3600) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except Exception:
                continue
            d = (chunk.get("choices") or [{}])[0].get("delta", {})
            piece = (d.get("content") or "") + (d.get("reasoning") or "") \
                    + (d.get("reasoning_content") or "")
            if piece:
                if t_first is None:
                    t_first = time.time()
                out.append(piece)
    return {"ttft_ms": (t_first - t0) * 1000 if t_first else None,
            "wall_ms": (time.time() - t0) * 1000,
            "text": "".join(out)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mlx", "gguf"])
    ap.add_argument("--model", default="qwen3.8-27b", choices=list(C.MODELS_CONFIG.keys()))
    ap.add_argument("--tier", default="agent", choices=T.TIER_ORDER)
    ap.add_argument("--draft-n", type=int, default=None)
    # Which MLX target to warm. Hardcoding mlx8 predated the quant sweep; the
    # candidate that has to be tested is whichever one is actually a contender.
    ap.add_argument("--quant", default="mlx8",
                    help="MLX quant key to serve (mlx arm only)")
    args = ap.parse_args()

    model_cfg = C.MODELS_CONFIG[args.model]
    mlx_cfg = C.MLX_MODELS_CONFIG.get(args.model, {})
    built = T.load_cached()
    draft_n = args.draft_n if args.draft_n is not None else (3 if args.arm == "mlx" else 3)

    arm_key = "mlx" if args.arm == "mlx" else "gguf-mtp"

    # mlx_vlm's Automatic Prompt Caching is OFF unless APC_ENABLED is set
    # (apc.py:3769 — `os.environ.get("APC_ENABLED", "0")`). llama.cpp caches prompt
    # prefixes by default, so without this the two arms are not comparable: MLX would
    # re-prefill the entire stable preamble on every turn while llama.cpp reuses it.
    # The first version of this benchmark made exactly that mistake.
    if args.arm == "mlx":
        os.environ["APC_ENABLED"] = "1"
        # Default capacity is BLOCK_SIZE(16) * NUM_BLOCKS(2048) = 32k tokens, which holds
        # the agent tier but not the deep tier; size it to the tier being measured.
        need_tokens = {"shallow": 4096, "agent": 32768, "deep": 98304}[args.tier]
        os.environ["APC_NUM_BLOCKS"] = str(need_tokens // 16)
        log(f"  APC_ENABLED=1  APC_NUM_BLOCKS={os.environ['APC_NUM_BLOCKS']} "
            f"({need_tokens} tokens capacity)")

    argv = server_argv(arm_key, model_cfg, mlx_cfg, draft_n, 1)
    model_name = (mlx_cfg.get("quants", {}).get(args.quant)
                  if args.arm == "mlx" else "qwen-local")
    if args.arm == "mlx" and not model_name:
        log(f"unknown MLX quant {args.quant!r} for {args.model}"); return 1

    srv = Server(argv, os.path.join(C.LOGS, f"warm_{args.arm}_{args.model}.log"))
    log(f"=== warm-cache: arm={args.arm} model={args.model} tier={args.tier} draft_n={draft_n} ===")
    if not srv.start():
        log("  !! server failed to start")
        srv.stop()
        return

    try:
        # Verify the cache is actually live rather than trusting the flag. A silently
        # disabled cache is indistinguishable from a slow engine in the TTFT numbers,
        # which is how the first run of this benchmark reached a wrong conclusion.
        if args.arm == "mlx":
            try:
                with urllib.request.urlopen(BASE + "/v1/cache/stats", timeout=10) as r:
                    stats = json.loads(r.read().decode())
                log(f"  /v1/cache/stats -> {json.dumps(stats)[:200]}")
                if not stats.get("enabled"):
                    log("  !! APC reports DISABLED — results would not be comparable; aborting")
                    return
            except Exception as e:
                log(f"  !! could not read /v1/cache/stats ({e}); aborting rather than "
                    f"reporting an unverified comparison")
                return

        # Fixed preamble = the tier prompt. Each turn appends the previous answer plus a
        # new short user message, so the stable prefix grows exactly as a session's does.
        base_user = T.user_text(built, "q1_model_selection", args.tier)
        messages = [{"role": "system", "content": QUESTIONS[0]["system"]},
                    {"role": "user", "content": base_user}]

        follow_ups = [
            "Summarise that in three bullet points.",
            "Which of those has the highest risk, and why?",
            "Give me a one-line action item.",
            "What would change your answer?",
        ]

        for turn in range(1, N_TURNS + 1):
            # llama.cpp: cache_prompt defaults on for the OpenAI route; set explicitly.
            r = one_turn(messages, model_name, GEN_TOKENS,
                         cache_prompt=True if args.arm == "gguf" else None)
            rec = {
                "ts": datetime.now(timezone.utc).isoformat(),
                "arm": args.arm, "model": args.model, "tier": args.tier,
                # Which build was actually served. Without this an MLX row is
                # ambiguous across four quants of the same model.
                "quant": args.quant if args.arm == "mlx" else "q8",
                "draft_n": draft_n, "turn": turn,
                "ttft_ms": r["ttft_ms"], "wall_ms": r["wall_ms"],
                "n_messages": len(messages),
            }
            with open(RESULTS_JSONL, "a") as f:
                f.write(json.dumps(rec) + "\n")
            log(f"  turn {turn}: TTFT {(r['ttft_ms'] or 0)/1000:6.2f}s  "
                f"wall {(r['wall_ms'] or 0)/1000:6.2f}s  ({len(messages)} messages)")

            messages = messages + [
                {"role": "assistant", "content": r["text"]},
                {"role": "user", "content": follow_ups[(turn - 1) % len(follow_ups)]},
            ]
    finally:
        srv.stop()
    log("=== warm-cache done ===")


if __name__ == "__main__":
    main()
