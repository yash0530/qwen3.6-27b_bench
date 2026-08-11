#!/usr/bin/env python3
"""Re-measure a cell that was already measured, to detect machine drift.

Why this exists: MLX's agent-tier prefill fell from 418 to 360 tok/s between two runs
whose prompts were byte-identical. Prefill cannot depend on `enable_thinking`, so either
the first figure was optimistic or the machine slowed after ~24h of sustained GPU load.
Cross-engine margins of a few percent are meaningless until that is settled.

This re-runs one GGUF cell (which no code change has touched) and compares against the
stored value. GGUF is the right probe precisely because its harness did not change: any
delta is the machine, not the benchmark.

Writes results/drift.jsonl. Does NOT touch results.jsonl.

  python3 drift_check.py --tier agent --draft-n 3
"""
import argparse
import json
import os
import statistics as st
import time
from datetime import datetime, timezone

import config as C
import tiers as T
from bench import Server, apply_template, run_generation, extract_timings, log
from questions import QUESTIONS

OUT = os.path.join(C.RESULTS, "drift.jsonl")


def stored(tier, draft_n, kv):
    vals = {"tok": [], "pf": []}
    with open(C.RESULTS_JSONL) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if (r.get("error") or r.get("phase") != "speed"
                    or r.get("runtime") not in (None, "gguf")
                    or (r.get("model") or "qwen3.6-27b") != "qwen3.6-27b"
                    or r.get("quant") != "q8" or r.get("draft_n") != draft_n
                    or (r.get("prompt_tier") or "shallow") != tier
                    or r.get("kv_quant") != kv):
                continue
            if r.get("predicted_per_second"):
                vals["tok"].append(r["predicted_per_second"])
            if r.get("prompt_per_second"):
                vals["pf"].append(r["prompt_per_second"])
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", default="agent")
    ap.add_argument("--draft-n", type=int, default=3)
    args = ap.parse_args()

    kv = None
    old = stored(args.tier, args.draft_n, kv)
    if not old["tok"]:
        log(f"no stored GGUF rows for tier={args.tier} draft_n={args.draft_n}; nothing to compare")
        return
    log(f"stored: {len(old['tok'])} runs, decode {st.mean(old['tok']):.2f} tok/s, "
        f"prefill {st.mean(old['pf']):.0f} tok/s")

    cfg = C.MODELS_CONFIG["qwen3.6-27b"]
    built = T.load_cached()
    ctx = C.ctx_for_tier(args.tier, C.SPEED_N_PREDICT)
    srv = Server("qwen3.6-27b", "q8", args.draft_n, cfg["quants"]["q8"],
                 cfg["reasoning_format"], ctx=ctx, kv_quant=kv, tier=args.tier)
    log(f"relaunching llama-server (ctx={ctx}) ...")
    if not srv.start():
        log("server failed"); srv.stop(); return

    now = {"tok": [], "pf": []}
    try:
        for q in QUESTIONS:
            msgs = [{"role": "system", "content": q["system"]},
                    {"role": "user", "content": T.user_text(built, q["id"], args.tier)}]
            prompt = apply_template(msgs)
            gen = run_generation(prompt, C.SPEED_N_PREDICT)
            tim = extract_timings(gen["final"])
            if tim.get("predicted_per_second"):
                now["tok"].append(tim["predicted_per_second"])
            if tim.get("prompt_per_second"):
                now["pf"].append(tim["prompt_per_second"])
            log(f"  {q['id']}: {tim.get('predicted_per_second'):.2f} tok/s, "
                f"prefill {tim.get('prompt_per_second'):.0f}")
    finally:
        srv.stop()

    d_tok = (st.mean(now["tok"]) - st.mean(old["tok"])) / st.mean(old["tok"]) * 100
    d_pf = (st.mean(now["pf"]) - st.mean(old["pf"])) / st.mean(old["pf"]) * 100
    rec = {"ts": datetime.now(timezone.utc).isoformat(), "tier": args.tier,
           "draft_n": args.draft_n, "kv": kv,
           "stored_tok": st.mean(old["tok"]), "now_tok": st.mean(now["tok"]),
           "stored_pf": st.mean(old["pf"]), "now_pf": st.mean(now["pf"]),
           "drift_tok_pct": d_tok, "drift_prefill_pct": d_pf}
    with open(OUT, "a") as f:
        f.write(json.dumps(rec) + "\n")
    log(f"DRIFT decode {d_tok:+.1f}%   prefill {d_pf:+.1f}%")
    log("  (|drift| under ~3% => machine stable, cross-engine margins are real)")


if __name__ == "__main__":
    main()
