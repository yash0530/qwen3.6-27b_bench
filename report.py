#!/usr/bin/env python3
"""Aggregate two-phase benchmark results, render charts, write REPORT.md + summary.json.

Speed metrics (tok/s, prompt speed, TTFT, MTP acceptance) come from phase="speed"
records (full grid, short cap). Token totals + answer text come from phase="full"
records (off config, 8192 cap). Judge scores (if present in results/judging/scores.json)
add a quality axis. Safe to re-run any time.
"""
import json
import os
import statistics as stats

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C

COLOR_PALETTE = [
    "#2563eb",  # Blue
    "#16a34a",  # Green
    "#dc2626",  # Red
    "#9333ea",  # Purple
    "#ea580c",  # Orange
    "#0d9488",  # Teal
    "#db2777",  # Pink
    "#4f46e5",  # Indigo
]


# ------------------------------------------------------------------- helper/labels
def get_label(model: str, quant: str) -> str:
    mname = model
    if model in C.MODELS_CONFIG:
        mname = C.MODELS_CONFIG[model]["name"]
    elif model in C.MLX_MODELS_CONFIG:
        mname = C.MLX_MODELS_CONFIG[model]["name"]
        
    qname = quant
    q_map = {
        "q5": "Q5", "q6": "Q6", "q8": "Q8",
        "mlx8": "MLX-8bit", "mlx6": "MLX-6bit", "mlx4": "MLX-4bit"
    }
    return f"{mname} ({q_map.get(quant, quant)})"


def get_color_map(model_quants):
    color_map = {}
    for i, mq in enumerate(model_quants):
        color_map[mq] = COLOR_PALETTE[i % len(COLOR_PALETTE)]
    return color_map


# ------------------------------------------------------------------- load/group
def load_all():
    recs = []
    if os.path.exists(C.RESULTS_JSON):
        with open(C.RESULTS_JSON) as f:
            recs = json.load(f)
    elif os.path.exists(C.RESULTS_JSONL):
        with open(C.RESULTS_JSONL) as f:
            for ln in f:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    
    res = []
    for r in recs:
        # Smoke rows are gate checks at a short token cap on a single question. They
        # would otherwise be averaged in as if they were sweep measurements.
        if r.get("error") is None and not r.get("smoke"):
            # normalize model
            if "model" not in r:
                r["model"] = "qwen3.6-27b"
            res.append(r)
    return res


def fmean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return stats.mean(xs) if xs else None


def fstd(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return stats.pstdev(xs) if len(xs) > 1 else 0.0


def load_scores():
    out = {}
    sp = os.path.join(C.JUDGING, "scores.json")
    if os.path.exists(sp):
        with open(sp) as f:
            data = json.load(f)
        for item in data.get("scores", []):
            model = item.get("model", "qwen3.6-27b")
            out.setdefault(model, {}).setdefault(item["quant"], {})[item["question_id"]] = item
    return out


# ----------------------------------------------------------------------- aggregate
def build_summary(speed, full, scores, mlx_speed=None, mlx_full=None):
    # Determine all unique combinations in speed
    gguf_combinations = sorted(list(set((r["model"], r["quant"]) for r in speed)))

    per_cell = []
    for model_id, quant in gguf_combinations:
        # get draft_ns for this model
        draft_ns = [0, 1, 2, 3, 4]
        if model_id in C.MODELS_CONFIG:
            draft_ns = C.MODELS_CONFIG[model_id]["draft_ns"]

        for dn in draft_ns:
            g = [r for r in speed if r["model"] == model_id and r["quant"] == quant and r["draft_n"] == dn]
            if not g:
                continue
            per_cell.append({
                "model": model_id, "quant": quant, "draft_n": dn,
                "config": "off" if dn == 0 else f"mtp{dn}",
                "n_runs": len(g),
                "tok_s_mean": fmean([r.get("predicted_per_second") for r in g]),
                "tok_s_std": fstd([r.get("predicted_per_second") for r in g]),
                "prompt_tok_s_mean": fmean([r.get("prompt_per_second") for r in g]),
                "ttft_ms_mean": fmean([r.get("ttft_ms") for r in g]),
                "acceptance_mean": fmean([r.get("acceptance_rate") for r in g]),
            })

    # full-length token totals per (model, quant)
    tokens = {}
    for model_id, quant in gguf_combinations:
        fg = [r for r in full if r["model"] == model_id and r["quant"] == quant]
        if fg:
            tokens[(model_id, quant)] = {
                "thinking_tokens_mean": fmean([r.get("thinking_tokens") for r in fg]),
                "answer_tokens_mean": fmean([r.get("answer_tokens") for r in fg]),
                "total_tokens_mean": fmean([r.get("predicted_n") for r in fg]),
                "answered": sum(1 for r in fg if (r.get("answer_tokens") or 0) > 0),
                "n": len(fg),
            }

    per_quant = []
    for model_id, quant in gguf_combinations:
        cells = [c for c in per_cell if c["model"] == model_id and c["quant"] == quant]
        if not cells:
            continue
        off = next((c for c in cells if c["draft_n"] == 0), None)
        mtp = [c for c in cells if c["draft_n"] > 0 and c["tok_s_mean"]]
        best = max(mtp, key=lambda c: c["tok_s_mean"]) if mtp else None
        off_tok = off["tok_s_mean"] if off else None
        
        q_scores = scores.get(model_id, {}).get(quant, {})
        overall = fmean([s.get("overall") for s in q_scores.values()]) if q_scores else None
        
        per_quant.append({
            "model": model_id, "quant": quant,
            "label": get_label(model_id, quant),
            "off_tok_s": off_tok,
            "best_draft_n": best["draft_n"] if best else None,
            "best_tok_s": best["tok_s_mean"] if best else None,
            "speedup_vs_off": (best["tok_s_mean"] / off_tok) if best and off_tok else None,
            "best_acceptance": best["acceptance_mean"] if best else None,
            "quality_overall": overall,
            **(tokens.get((model_id, quant), {})),
        })

    # MLX models. These now carry a real MTP sweep (draft_block_size) plus tier and KV
    # axes, so they aggregate exactly like the GGUF arm rather than collapsing to a
    # single number. The previous version averaged every MLX record together and
    # hard-coded best_draft_n/best_acceptance to None, which made it structurally
    # impossible for MLX to show a speedup even when it had one.
    mlx_cells = []
    mlx_combinations = sorted(set((r["model"], r["quant"]) for r in (mlx_speed or [])))
    for model_id, quant in mlx_combinations:
        sg_all = [r for r in mlx_speed if r["model"] == model_id and r["quant"] == quant]
        keys = sorted(set((r.get("prompt_tier", "shallow"), r.get("kv_bits"),
                           r.get("draft_n") or 0) for r in sg_all),
                      key=lambda k: (k[0], str(k[1]), k[2]))
        for tier, kvb, dn in keys:
            g = [r for r in sg_all if r.get("prompt_tier", "shallow") == tier
                 and r.get("kv_bits") == kvb and (r.get("draft_n") or 0) == dn]
            if not g:
                continue
            mlx_cells.append({
                "model": model_id, "quant": quant, "runtime": "mlx",
                "prompt_tier": tier, "kv_bits": kvb, "draft_n": dn,
                "config": "off" if dn == 0 else f"mtp{dn}",
                "n_runs": len(g),
                "tok_s_mean": fmean([r.get("predicted_per_second") for r in g]),
                "tok_s_std": fstd([r.get("predicted_per_second") for r in g]),
                "prompt_tok_s_mean": fmean([r.get("prompt_per_second") for r in g]),
                "ttft_ms_mean": fmean([r.get("ttft_ms") for r in g]),
                "acceptance_mean": fmean([r.get("acceptance_rate") for r in g]),
                "peak_memory_gb_mean": fmean([r.get("peak_memory_gb") for r in g]),
            })

        # Headline slice: shallow prompt, unquantized KV — the configuration the existing
        # GGUF numbers were measured under, so the top-line comparison is like-for-like.
        head = [c for c in mlx_cells if c["model"] == model_id and c["quant"] == quant
                and c["prompt_tier"] == "shallow" and c["kv_bits"] is None]
        if not head:
            head = [c for c in mlx_cells if c["model"] == model_id and c["quant"] == quant]
        off = next((c for c in head if c["draft_n"] == 0), None)
        mtp = [c for c in head if c["draft_n"] > 0 and c["tok_s_mean"]]
        best = max(mtp, key=lambda c: c["tok_s_mean"]) if mtp else None
        off_tok = off["tok_s_mean"] if off else None

        fg = [r for r in (mlx_full or []) if r["model"] == model_id and r["quant"] == quant]
        q_scores = scores.get(model_id, {}).get(quant, {})
        overall = fmean([s.get("overall") for s in q_scores.values()]) if q_scores else None
        per_quant.append({
            "model": model_id, "quant": quant, "label": get_label(model_id, quant),
            "runtime": "mlx",
            "off_tok_s": off_tok,
            "best_draft_n": best["draft_n"] if best else None,
            "best_tok_s": best["tok_s_mean"] if best else off_tok,
            "speedup_vs_off": (best["tok_s_mean"] / off_tok) if best and off_tok else None,
            "best_acceptance": best["acceptance_mean"] if best else None,
            "quality_overall": overall,
            "thinking_tokens_mean": fmean([r.get("thinking_tokens") for r in fg]),
            "answer_tokens_mean": fmean([r.get("answer_tokens") for r in fg]),
            "total_tokens_mean": fmean([r.get("predicted_n") for r in fg]),
            "ttft_ms_mean": off["ttft_ms_mean"] if off else None,
            "prompt_tok_s_mean": off["prompt_tok_s_mean"] if off else None,
            "n": len(fg), "answered": sum(1 for r in fg if (r.get("answer_tokens") or 0) > 0),
        })

    summary = {
        "meta": {
            "seed": C.SEED, "temp": C.TEMP,
            "top_p": C.TOP_P, "top_k": C.TOP_K, "speed_cap": C.SPEED_N_PREDICT,
            "full_cap": C.FULL_N_PREDICT, "ctx": C.CTX,
            "n_speed": len(speed), "n_full": len(full),
        },
        "per_quant_config": per_cell,
        "per_quant": per_quant,
        # MLX cells are kept separate from per_quant_config because they carry two extra
        # axes (prompt_tier, kv_bits) that the GGUF cells don't. The depth-tier table
        # reads from here.
        "mlx_cells": mlx_cells,
        "determinism": determinism_check(speed, mlx_speed),
    }
    with open(C.SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def _read_jsonl(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return []
    out = []
    with open(p) as f:
        for ln in f:
            try:
                out.append(json.loads(ln))
            except Exception:
                pass
    return out


def _aux_sections():
    """Warm-cache, concurrency and append-only tables.

    These live in their own jsonl files rather than results.jsonl because they measure
    whole sessions and client fan-out, not single generations. They are the measurements
    that actually decided the recommendation, so they belong in the report rather than
    only in the raw data.
    """
    L = []

    warm = _read_jsonl("warmcache.jsonl")
    if warm:
        by = {}
        for r in warm:
            by.setdefault((r["model"], r["arm"]), {})[r["turn"]] = r["ttft_ms"] / 1000
        L.append("## Warm-cache TTFT (multi-turn session at agent depth)\n")
        L.append("_The deciding measurement. An agent loop re-sends a large stable preamble "
                 "every turn, so what matters is TTFT on turns 2+, not the cold first turn._\n")
        L.append("| model | engine | cold (t1) | warm mean (t2-t5) | reduction |")
        L.append("|---|---|---|---|---|")
        for k in sorted(by):
            t = by[k]
            w = [t[i] for i in sorted(t) if i > 1]
            if not w or 1 not in t:
                continue
            wm = sum(w) / len(w)
            L.append(f"| {get_label(k[0], 'q8' if k[1] == 'gguf' else 'mlx8')} | {k[1]} | "
                     f"{t[1]:.1f} s | **{wm:.2f} s** | {(1 - wm / t[1]) * 100:.0f}% |")
        L.append("")

    ao = _read_jsonl("append_only.jsonl")
    if ao:
        L.append("## Append-only continuation (MLX)\n")
        L.append("_mlx_vlm.server re-renders the chat template each request, which forces a "
                 "cache rewind Qwen 3.6's hybrid cache cannot do. Appending to the token "
                 "sequence the cache already holds needs no rewind — so MLX's warm-cache "
                 "deficit is a serving-layer gap, not an engine limitation._\n")
        L.append("| model | cold TTFT | warm TTFT | speedup | output preserved |")
        L.append("|---|---|---|---|---|")
        for r in ao:
            L.append(f"| {get_label(r['model'], 'mlx8')} | {r['ttft_cold_s']:.2f} s | "
                     f"**{r['ttft_warm_s']:.2f} s** | {(r['speedup'] or 0):.0f}x | "
                     f"{'yes' if r['output_preserved'] else 'NO (see notes)'} |")
        L.append("")

    conc = _read_jsonl("concurrency.jsonl")
    if conc:
        by = {}
        for r in conc:
            by[(r["model"], r["arm"], r["concurrency"])] = r
        models = sorted({k[0] for k in by})
        arms = ["mlx", "gguf-mtp", "gguf-batch"]
        levels = sorted({k[2] for k in by})
        L.append("## Concurrency (shallow prompts, aggregate chars/s | mean TTFT)\n")
        L.append("_`gguf-mtp` is `-np 1` + MTP (requests queue; what llm-serve runs). "
                 "`gguf-batch` is `-np N` with MTP off. llama.cpp cannot do both._\n")
        for m in models:
            L.append(f"**{m}**\n")
            L.append("| arm | " + " | ".join(f"c={c}" for c in levels) + " |")
            L.append("|---" * (len(levels) + 1) + "|")
            for a in arms:
                cells = []
                for c in levels:
                    r = by.get((m, a, c))
                    cells.append(f"{(r['aggregate_chars_per_s'] or 0):.0f} · "
                                 f"{(r['ttft_ms_mean'] or 0) / 1000:.1f}s" if r else "—")
                L.append(f"| {a} | " + " | ".join(cells) + " |")
            L.append("")

    drift = _read_jsonl("drift.jsonl")
    if drift:
        d = drift[-1]
        L.append("## Machine drift control\n")
        L.append(f"- Re-measured an unchanged GGUF cell (tier `{d['tier']}`, draft-n "
                 f"{d['draft_n']}): decode **{d['drift_tok_pct']:+.1f}%**, prefill "
                 f"**{d['drift_prefill_pct']:+.1f}%** versus the stored value. Cross-engine "
                 f"margins below ~5% are unresolved.\n")

    return L


def determinism_check(speed, mlx_speed=None):
    """Does speculative decoding change the output under a fixed seed?

    Exact rejection sampling is supposed to make MTP output-preserving, so any divergence
    is a correctness bug — and speed bought by changing the output is not speed worth
    having. Both runtimes are probed, each against its own MTP-off baseline.

    The comparison cell includes tier and KV setting: the same question at a different
    prompt depth or KV precision is a different generation, and comparing across those
    would manufacture false divergences.
    """
    def cell(r):
        return (r.get("model"), r.get("quant"), r.get("question_id"),
                r.get("prompt_tier", "shallow"),
                r.get("kv_bits") if r.get("runtime") == "mlx_vlm" else r.get("kv_quant"))

    def probe(recs):
        base, total, diff = {}, 0, 0
        for r in recs:
            if (r.get("draft_n") or 0) == 0:
                base[cell(r)] = r.get("output_sha256")
        for r in recs:
            if (r.get("draft_n") or 0) > 0:
                b = base.get(cell(r))
                if b is not None and r.get("output_sha256") is not None:
                    total += 1
                    if r["output_sha256"] != b:
                        diff += 1
        return total, diff

    g_total, g_diff = probe(speed)
    m_total, m_diff = probe(mlx_speed or [])
    return {
        "mtp_vs_off_compared": g_total + m_total,
        "mtp_vs_off_diverged": g_diff + m_diff,
        "gguf_compared": g_total, "gguf_diverged": g_diff,
        "mlx_compared": m_total, "mlx_diverged": m_diff,
    }


# --------------------------------------------------------------------------- charts
def chart_tok_s(summary):
    plt.figure(figsize=(9, 5.5))
    gguf_runs = [q for q in summary["per_quant"] if q.get("runtime") != "mlx"]
    model_quants = [(q["model"], q["quant"]) for q in gguf_runs]
    color_map = get_color_map(model_quants)

    for q in gguf_runs:
        model_id = q["model"]
        quant = q["quant"]
        draft_ns = [0, 1, 2, 3, 4]
        if model_id in C.MODELS_CONFIG:
            draft_ns = C.MODELS_CONFIG[model_id]["draft_ns"]

        xs, ys, es = [], [], []
        for dn in draft_ns:
            c = next((c for c in summary["per_quant_config"]
                      if c["model"] == model_id and c["quant"] == quant and c["draft_n"] == dn), None)
            if c and c["tok_s_mean"]:
                xs.append(dn); ys.append(c["tok_s_mean"]); es.append(c["tok_s_std"])
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                         color=color_map[(model_id, quant)], label=q["label"])
    plt.xlabel("MTP draft tokens (--spec-draft-n-max)")
    plt.ylabel("Decode speed (tokens/sec)")
    plt.title("Decode throughput vs MTP draft depth (mean over questions)")
    plt.grid(True, alpha=0.3); plt.legend()
    _save("01_decode_tok_s.png")


def chart_speedup(summary):
    plt.figure(figsize=(9, 5.5))
    gguf_runs = [q for q in summary["per_quant"] if q.get("runtime") != "mlx"]
    model_quants = [(q["model"], q["quant"]) for q in gguf_runs]
    color_map = get_color_map(model_quants)

    for q in gguf_runs:
        model_id = q["model"]
        quant = q["quant"]
        off = next((c for c in summary["per_quant_config"]
                    if c["model"] == model_id and c["quant"] == quant and c["draft_n"] == 0), None)
        if not off or not off["tok_s_mean"]:
            continue
        draft_ns = [0, 1, 2, 3, 4]
        if model_id in C.MODELS_CONFIG:
            draft_ns = C.MODELS_CONFIG[model_id]["draft_ns"]
        xs, ys = [], []
        for dn in draft_ns:
            if dn == 0:
                continue
            c = next((c for c in summary["per_quant_config"]
                      if c["model"] == model_id and c["quant"] == quant and c["draft_n"] == dn), None)
            if c and c["tok_s_mean"]:
                xs.append(dn); ys.append(c["tok_s_mean"] / off["tok_s_mean"])
        if xs:
            plt.plot(xs, ys, marker="o", color=color_map[(model_id, quant)], label=q["label"])
    plt.axhline(1.0, color="gray", ls="--", alpha=0.6, label="MTP off")
    plt.xlabel("MTP draft tokens"); plt.ylabel("Speedup vs MTP off (x)")
    plt.title("MTP speedup over baseline"); plt.grid(True, alpha=0.3); plt.legend()
    _save("02_speedup.png")


def chart_acceptance(summary):
    plt.figure(figsize=(9, 5.5))
    gguf_runs = [q for q in summary["per_quant"] if q.get("runtime") != "mlx"]
    model_quants = [(q["model"], q["quant"]) for q in gguf_runs]
    color_map = get_color_map(model_quants)

    for q in gguf_runs:
        model_id = q["model"]
        quant = q["quant"]
        draft_ns = [0, 1, 2, 3, 4]
        if model_id in C.MODELS_CONFIG:
            draft_ns = C.MODELS_CONFIG[model_id]["draft_ns"]
        xs, ys = [], []
        for dn in draft_ns:
            if dn == 0:
                continue
            c = next((c for c in summary["per_quant_config"]
                      if c["model"] == model_id and c["quant"] == quant and c["draft_n"] == dn), None)
            if c and c["acceptance_mean"] is not None:
                xs.append(dn); ys.append(c["acceptance_mean"] * 100)
        if xs:
            plt.plot(xs, ys, marker="o", color=color_map[(model_id, quant)], label=q["label"])
    plt.xlabel("MTP draft tokens"); plt.ylabel("Draft acceptance (%)")
    plt.title("MTP draft acceptance rate"); plt.grid(True, alpha=0.3); plt.legend()
    _save("03_acceptance.png")


def chart_prompt_speed(summary):
    plt.figure(figsize=(8, 5))
    all_runs = summary["per_quant"]
    model_quants = [(q["model"], q["quant"]) for q in all_runs]
    color_map = get_color_map(model_quants)

    labels = [q["label"] for q in all_runs]
    vals = [q.get("prompt_tok_s_mean") or 0 for q in all_runs]
    colors = [color_map[(q["model"], q["quant"])] for q in all_runs]

    plt.bar(labels, vals, color=colors)
    plt.ylabel("Prompt processing (tokens/sec)")
    plt.title("Prompt processing speed by model & quant")
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    _save("04_prompt_speed.png")


def chart_ttft(summary):
    plt.figure(figsize=(9, 5.5))
    gguf_runs = [q for q in summary["per_quant"] if q.get("runtime") != "mlx"]
    model_quants = [(q["model"], q["quant"]) for q in gguf_runs]
    color_map = get_color_map(model_quants)

    for q in gguf_runs:
        model_id = q["model"]
        quant = q["quant"]
        draft_ns = [0, 1, 2, 3, 4]
        if model_id in C.MODELS_CONFIG:
            draft_ns = C.MODELS_CONFIG[model_id]["draft_ns"]
        xs, ys = [], []
        for dn in draft_ns:
            c = next((c for c in summary["per_quant_config"]
                      if c["model"] == model_id and c["quant"] == quant and c["draft_n"] == dn), None)
            if c and c["ttft_ms_mean"]:
                xs.append(dn); ys.append(c["ttft_ms_mean"])
        if xs:
            plt.plot(xs, ys, marker="o", color=color_map[(model_id, quant)], label=q["label"])
    plt.xlabel("MTP draft tokens"); plt.ylabel("Time to first token (ms)")
    plt.title("Latency to first token"); plt.grid(True, alpha=0.3); plt.legend()
    _save("05_ttft.png")


def chart_tokens(summary):
    all_runs = summary["per_quant"]
    x = np.arange(len(all_runs)); w = 0.6
    think = [(q.get("thinking_tokens_mean") or 0) for q in all_runs]
    ans = [(q.get("answer_tokens_mean") or 0) for q in all_runs]
    plt.figure(figsize=(9, 5.5))
    plt.bar(x, think, w, label="thinking tokens", color="#9333ea")
    plt.bar(x, ans, w, bottom=think, label="answer tokens", color="#f59e0b")
    plt.xticks(x, [q["label"] for q in all_runs], rotation=15, ha="right")
    plt.ylabel("Mean tokens per answer (full cap)")
    plt.title("Thinking vs answer token volume by model & quant")
    plt.legend()
    _save("06_tokens.png")


def chart_quality(summary):
    pq = [q for q in summary["per_quant"] if q.get("quality_overall") is not None]
    if not pq:
        return False
    plt.figure(figsize=(8, 5))
    model_quants = [(q["model"], q["quant"]) for q in pq]
    color_map = get_color_map(model_quants)
    colors = [color_map[(q["model"], q["quant"])] for q in pq]

    plt.bar([q["label"] for q in pq], [q["quality_overall"] for q in pq], color=colors)
    plt.ylim(0, 10); plt.ylabel("Judge quality (1-10)")
    plt.title("Answer quality by model & quant (pluggable judge)")
    plt.xticks(rotation=15, ha="right")
    for i, q in enumerate(pq):
        plt.text(i, q["quality_overall"], f"{q['quality_overall']:.1f}", ha="center", va="bottom")
    _save("07_quality.png")
    return True


def chart_quality_speed(summary):
    pq = [q for q in summary["per_quant"]
          if q.get("quality_overall") is not None and q.get("best_tok_s")]
    if not pq:
        return False
    plt.figure(figsize=(8.5, 6))
    model_quants = [(q["model"], q["quant"]) for q in pq]
    color_map = get_color_map(model_quants)

    for q in pq:
        plt.scatter(q["best_tok_s"], q["quality_overall"], s=160,
                    color=color_map[(q["model"], q["quant"])], zorder=3)
        ntag = f"n={q['best_draft_n']}" if q.get("best_draft_n") is not None else "no MTP"
        plt.annotate(f"{q['label']} ({ntag})",
                     (q["best_tok_s"], q["quality_overall"]),
                     textcoords="offset points", xytext=(8, 6))
    plt.xlabel("Best decode speed (tok/s)")
    plt.ylabel("Judge quality (1-10)")
    plt.title("Quality vs speed trade-off (top-right wins)")
    plt.grid(True, alpha=0.3)
    _save("08_quality_vs_speed.png")
    return True


def chart_runtime_compare(summary):
    pq = [q for q in summary["per_quant"] if q.get("best_tok_s")]
    if not pq:
        return
    plt.figure(figsize=(9, 5.5))
    model_quants = [(q["model"], q["quant"]) for q in pq]
    color_map = get_color_map(model_quants)

    labels, vals, cols = [], [], []
    for q in pq:
        tag = f"\n(n={q['best_draft_n']})" if q.get("best_draft_n") is not None else "\n(no MTP)"
        labels.append(q["label"] + tag)
        vals.append(q["best_tok_s"])
        cols.append(color_map[(q["model"], q["quant"])])
    plt.bar(labels, vals, color=cols)
    plt.ylabel("Best decode speed (tokens/sec)")
    plt.title("Runtime comparison: GGUF vs MLX")
    plt.xticks(rotation=15, ha="right")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
    _save("09_runtime_tok_s.png")


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS, name), dpi=130)
    plt.close()


# --------------------------------------------------------------------------- report
def write_report(summary, has_quality, has_mlx=False):
    m = summary["meta"]
    L = []
    L.append("# Local LLM Benchmarks — Multi-Model Analysis\n")
    L.append(f"_Speed sweep: {m['n_speed']} runs · "
             f"full-length: {m['n_full']} runs · "
             f"temp {m['temp']}, seed {m['seed']}, ctx {m['ctx']}._\n")

    L.append("## TL;DR\n")
    for q in summary["per_quant"]:
        qual = f"{q['quality_overall']:.1f}/10" if q.get("quality_overall") is not None else "ungraded"
        if q.get("runtime") == "mlx":
            L.append(f"- **{q['label']}** (MLX) — **{(q['best_tok_s'] or 0):.1f} tok/s**, "
                     f"quality {qual}, ~{(q.get('total_tokens_mean') or 0):.0f} tok/answer.")
        else:
            spd = f"{q['speedup_vs_off']:.2f}x" if q.get("speedup_vs_off") else "n/a"
            ntag = f"draft-n={q['best_draft_n']}" if q.get("best_draft_n") is not None else "no MTP"
            acc_s = f" ({(q['best_acceptance'] or 0)*100:.0f}% accept)" if q.get("best_acceptance") is not None else ""
            spd_s = f" ({spd} vs off)" if q.get("speedup_vs_off") else ""
            L.append(f"- **{q['label']}** — peak **{(q['best_tok_s'] or 0):.1f} tok/s** at "
                     f"{ntag}{spd_s}{acc_s}; quality {qual}; "
                     f"~{(q.get('total_tokens_mean') or 0):.0f} tok/answer.")
    L.append("")

    L.append("## Charts\n")
    cfiles = ["01_decode_tok_s.png", "02_speedup.png", "03_acceptance.png",
              "04_prompt_speed.png", "05_ttft.png", "06_tokens.png"]
    if has_quality:
        cfiles += ["07_quality.png", "08_quality_vs_speed.png"]
    if has_mlx:
        cfiles += ["09_runtime_tok_s.png"]
    for cf in cfiles:
        L.append(f"![{cf}](results/charts/{cf})\n")

    L.append("## Speed sweep (mean over questions)\n")
    L.append("| model/quant | draft-n | tok/s | ±sd | accept % | prompt tok/s | TTFT ms |")
    L.append("|---|---|---|---|---|---|---|")
    for c in summary["per_quant_config"]:
        acc = f"{c['acceptance_mean']*100:.0f}" if c["acceptance_mean"] is not None else "-"
        lbl = get_label(c["model"], c["quant"])
        L.append(f"| {lbl} | {c['config']} | {(c['tok_s_mean'] or 0):.1f} | "
                 f"{(c['tok_s_std'] or 0):.1f} | {acc} | {(c['prompt_tok_s_mean'] or 0):.0f} | "
                 f"{(c['ttft_ms_mean'] or 0):.0f} |")
    L.append("")

    cells = summary.get("mlx_cells") or []
    if cells:
        L.append("## MLX by prompt depth and KV precision\n")
        L.append("_Decode rate at ~200 tok (`shallow`) does not predict the agent loop, which "
                 "sends ~23k tokens before the user types. `kv` is the KV-cache precision: "
                 "`fp16` unquantized, `8` matching llama.cpp's `-ctk q8_0 -ctv q8_0`._\n")
        L.append("| model/quant | tier | kv | config | tok/s | ±sd | accept % | prefill tok/s | TTFT ms | peak GB |")
        L.append("|---|---|---|---|---|---|---|---|---|---|")
        for c in cells:
            acc = f"{c['acceptance_mean']*100:.0f}" if c.get("acceptance_mean") is not None else "-"
            L.append(f"| {get_label(c['model'], c['quant'])} | {c['prompt_tier']} | "
                     f"{c['kv_bits'] or 'fp16'} | {c['config']} | {(c['tok_s_mean'] or 0):.1f} | "
                     f"{(c['tok_s_std'] or 0):.1f} | {acc} | {(c['prompt_tok_s_mean'] or 0):.0f} | "
                     f"{(c['ttft_ms_mean'] or 0):.0f} | {(c.get('peak_memory_gb_mean') or 0):.1f} |")
        L.append("")

    L.append("## Full-length output (8192 cap, off config)\n")
    L.append("| model/quant | total tok | thinking tok | answer tok | answers completed |")
    L.append("|---|---|---|---|---|")
    for q in summary["per_quant"]:
        L.append(f"| {q['label']} | {(q.get('total_tokens_mean') or 0):.0f} | "
                 f"{(q.get('thinking_tokens_mean') or 0):.0f} | "
                 f"{(q.get('answer_tokens_mean') or 0):.0f} | "
                 f"{q.get('answered', 0)}/{q.get('n', 0)} |")
    L.append("")

    d = summary["determinism"]
    L.append("## Output determinism (MTP correctness probe, #23302)\n")
    if d.get("mlx_compared"):
        L.append(f"- llama.cpp: {d['gguf_diverged']}/{d['gguf_compared']} diverged · "
                 f"MLX: {d['mlx_diverged']}/{d['mlx_compared']} diverged\n")
    L.append(f"- {d['mtp_vs_off_diverged']}/{d['mtp_vs_off_compared']} MTP runs produced a "
             f"different output than their MTP-off baseline (fixed seed). 0 = MTP is "
             f"output-preserving on this build; >0 flags the known determinism bug.\n")

    L.extend(_aux_sections())

    L.append("## Measurement notes and known limitations\n")
    L.append("- **Arm parity is gated by `validate_parity.py`.** Run it before trusting any "
             "GGUF-vs-MLX number here. It exists because an earlier revision compared the two "
             "arms for ~250 runs while they were doing *different work*: `enable_thinking` was "
             "passed to `stream_generate` but not to `apply_chat_template`, so the Qwen3.6 "
             "template emitted a pre-closed `<think></think>` block and the MLX arm never "
             "reasoned, while llama.cpp under `--jinja` reasoned throughout. Both arms ran "
             "green and produced plausible tok/s. The checker measures reasoning rate per arm "
             "(95% vs 0% at the time) and fails on a gap over 15%, along with prompt-length "
             "parity, empty answers, missing acceptance, and cache leakage.")
    L.append("- **Acted on, and verified end-to-end.** The fp16-KV finding below was applied to "
             "`local-setup` (`llm-serve`, branch `tune/kv-fp16-and-draft-depth`) and re-measured "
             "on the live server with the 35B at 23k depth: RSS 38.7 -> 41.0 GB (50 GB wired "
             "limit), cold TTFT 28.90 -> 24.01 s, decode +11.6%, warm turn 5.27 -> 4.76 s. The "
             "27B draft depth was raised 2 -> 3 in the same change.")
    L.append("- **MLX does not reuse prompt prefixes for Qwen 3.6; llama.cpp does.** This is the "
             "single most decision-relevant finding, because an agent loop re-sends a large stable "
             "preamble every turn. Measured over a 5-turn session at 23k depth: llama.cpp warm TTFT "
             "**1.68 s** (27B) / **0.57 s** (35B) against cold 85 s / 24.9 s, i.e. near-perfect reuse. "
             "MLX stays flat at ~55.5-56.6 s (27B) and ~12.6 s (35B) across all turns.")
    L.append("  - mlx_vlm's Automatic Prompt Caching is **off unless `APC_ENABLED=1`** "
             "(`apc.py:3769`). Enabling it — verified live via `/v1/cache/stats` — changed nothing: "
             "`lookups_hit=0, lookups_miss=0, stores=0`.")
    L.append("  - Probable cause is architectural. Qwen 3.6 is hybrid: "
             "`qwen3_5/language.py:2615` returns `ArraysCache` for Gated Delta Network layers and "
             "`KVCache` for attention layers. Block-mode APC requires *every* entry to be a "
             "`KVCache` (`apc.py:292`), so it cannot apply. The 'exact' whole-prefix fallback "
             "nominally accepts `ArraysCache` but stores nothing in practice. llama.cpp's slot "
             "cache does longest-common-prefix reuse, which survives a growing conversation; "
             "exact whole-prefix snapshots do not.")
    L.append("- **MLX's continuous batching underperforms under load.** At 10 concurrent clients "
             "llama.cpp `-np N` (MTP off) beats MLX on aggregate throughput, mean TTFT and max "
             "latency for both models; on the 27B MLX's max latency is 644 s vs 193 s. So while "
             "mlx_vlm *can* batch and speculate simultaneously — which llama.cpp cannot — the "
             "combination does not pay off here.")
    L.append("- **Machine drift over the measurement window was -4.5% decode / -5.7% prefill** "
             "(`drift_check.py`, re-measuring an unchanged GGUF cell). Cross-engine margins below "
             "~5% should be treated as unresolved.")
    L.append("- **Quantized KV + MTP is broken in mlx_vlm 0.6.3 at depth.** "
             "`models/qwen3_5/language.py:1481` does `keys.shape[-2]` on the KV cache, but a "
             "quantized MLX cache is a *list* of (values, scales, biases), so the "
             "speculative-verify path raises `AttributeError: 'list' object has no attribute "
             "'shape'`. Measured: shallow+q8+MTP passes, agent+fp16+MTP passes, agent+q8+no-MTP "
             "passes, agent+q8+MTP fails at every block size. `turboquant` fails differently.")
    L.append("- Consequently the MLX q8 arm is labelled **q8 decode-only**: quantization is "
             "deferred past the prompt (`quantized_kv_start`), so the prompt's KV stays "
             "unquantized. That runs, but it does **not** deliver q8 KV's memory saving at long "
             "context and is therefore *not* equivalent to llama.cpp's `-ctk q8_0 -ctv q8_0`. "
             "There is currently no MLX equivalent of the llm-serve config.")
    L.append("- **MLX draft-block-size 1 is a silent no-op**: the drafter proposes "
             "`block_size - 1` tokens, so 1 proposes none, emits a single token and stops — "
             "reporting an absurd tok/s. Valid speculative depths start at 2.")
    L.append("- `prefill_step_size` is pinned at 2048. Raising it above the prompt length to "
             "avoid chunked prefill OOMs Metal at 23k tokens.")
    L.append("- Legacy `mlx_lm.server` records are excluded from this report: they had no "
             "speculative decoding, a corrupted thinking/answer split, and TTFT-derived prompt "
             "speed. They remain in results.jsonl for provenance only.")
    L.append("- Prompt text is byte-identical across runtimes (built once, cached to "
             "`results/prompt_tiers.json`), so both engines tokenize the same bytes at each "
             "depth tier rather than being compared on prompts that differ by a few tokens.\n")

    best = _pick_overall(summary)
    if best:
        L.append("## Hosting recommendation\n")
        m_id = best["model"]
        # Emit a launch line for whichever runtime actually won. The previous version
        # looked the winner up in MODELS_CONFIG (GGUF only), so an MLX config could never
        # be recommended no matter how it scored — the comparison was decided in advance.
        if best.get("runtime") == "mlx":
            m_cfg = C.MLX_MODELS_CONFIG.get(m_id, {})
            model_path = (m_cfg.get("quants") or {}).get(best["quant"], "<model>")
            draft = m_cfg.get("draft_model")
            bs = best.get("best_draft_n")
            L.append(f"On this machine, serve **{best['label']}** with peak speed. Launch line:\n")
            L.append("```bash")
            L.append(f"mlx_vlm.server --model {model_path} \\")
            if draft and bs:
                L.append(f"  --draft-model {draft} --draft-kind mtp --draft-block-size {bs} \\")
            L.append(f"  --kv-bits 8 --kv-group-size {C.KV_GROUP_SIZE} "
                     f"--kv-quant-scheme {C.KV_QUANT_SCHEME} \\")
            L.append("  --host 127.0.0.1 --port 8080")
            L.append("```\n")
        else:
            m_cfg = C.MODELS_CONFIG.get(m_id)
            if m_cfg:
                model_path = m_cfg["quants"][best["quant"]]
                dn_args = f"--spec-type draft-mtp --spec-draft-n-max {best['best_draft_n']}" if best.get("best_draft_n") else "--spec-type none"
                rf_arg = f" --reasoning-format {m_cfg['reasoning_format']}" if m_cfg.get("reasoning_format") and m_cfg["reasoning_format"] != "none" else ""
                L.append(f"On this machine, serve **{best['label']}** with peak speed. Launch line:\n")
                L.append("```bash")
                L.append(f"llama-server -m {model_path} \\")
                L.append(f"  {dn_args} \\")
                L.append(f"  -c 16384 -ngl 99 -fa on -np 1 --jinja{rf_arg} \\")
                L.append("  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080")
                L.append("```\n")

    with open(os.path.join(C.REPO, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote REPORT.md")


def write_html_report(summary, has_quality, has_mlx=False):
    m = summary["meta"]
    
    tldr_cards = []
    for q in summary["per_quant"]:
        qual = f"{q['quality_overall']:.1f}/10" if q.get("quality_overall") is not None else "ungraded"
        if q.get("runtime") == "mlx":
            tldr_cards.append(f"""
      <div class="card">
        <div class="card-title"><span class="badge badge-accent">MLX</span> {q['label']}</div>
        <div class="card-stat">{(q['best_tok_s'] or 0):.1f} <span style="font-size: 1rem; font-weight: normal; color: var(--text-muted);">tok/s</span></div>
        <div class="card-detail">
          Quality: <strong>{qual}</strong><br>
          Avg response size: <strong>{(q.get('total_tokens_mean') or 0):.0f} tokens</strong>
        </div>
      </div>""")
        else:
            spd = f"{q['speedup_vs_off']:.2f}x" if q.get("speedup_vs_off") else "n/a"
            ntag = f"draft-n={q['best_draft_n']}" if q.get("best_draft_n") is not None else "no MTP"
            acc_s = f" ({q['best_acceptance']*100:.0f}% accept)" if q.get("best_acceptance") is not None else ""
            tldr_cards.append(f"""
      <div class="card">
        <div class="card-title"><span class="badge badge-primary">GGUF</span> {q['label']}</div>
        <div class="card-stat">{(q['best_tok_s'] or 0):.1f} <span style="font-size: 1rem; font-weight: normal; color: var(--text-muted);">tok/s</span></div>
        <div class="card-detail">
          Peak config: <strong>{ntag}</strong> ({spd} speedup vs off){acc_s}<br>
          Quality: <strong>{qual}</strong><br>
          Avg response size: <strong>{(q.get('total_tokens_mean') or 0):.0f} tokens</strong>
        </div>
      </div>""")
    
    tldr_html = "\n".join(tldr_cards)
    
    cfiles = ["01_decode_tok_s.png", "02_speedup.png", "03_acceptance.png",
              "04_prompt_speed.png", "05_ttft.png", "06_tokens.png"]
    if has_quality:
        cfiles += ["07_quality.png", "08_quality_vs_speed.png"]
    if has_mlx:
        cfiles += ["09_runtime_tok_s.png"]
        
    charts_cards = []
    for cf in cfiles:
        title = cf.replace("_", " ").replace(".png", "").capitalize()
        if title[:2].isdigit():
            title = title[3:]
        charts_cards.append(f"""
      <div class="chart-box">
        <img src="results/charts/{cf}" alt="{title}">
        <div class="chart-title">{title}</div>
      </div>""")
    charts_html = "\n".join(charts_cards)
    
    speed_rows = []
    for c in summary["per_quant_config"]:
        acc = f"{c['acceptance_mean']*100:.0f}%" if c["acceptance_mean"] is not None else "-"
        lbl = get_label(c["model"], c["quant"])
        speed_rows.append(f"""
        <tr>
          <td><strong>{lbl}</strong></td>
          <td><span class="badge badge-primary">{c['config']}</span></td>
          <td><strong>{(c['tok_s_mean'] or 0):.1f}</strong></td>
          <td>±{(c['tok_s_std'] or 0):.1f}</td>
          <td>{acc}</td>
          <td>{(c['prompt_tok_s_mean'] or 0):.0f}</td>
          <td>{(c['ttft_ms_mean'] or 0):.0f} ms</td>
        </tr>""")
    speed_table_rows = "\n".join(speed_rows)
    
    full_rows = []
    for q in summary["per_quant"]:
        full_rows.append(f"""
        <tr>
          <td><strong>{q['label']}</strong></td>
          <td><strong>{(q.get('total_tokens_mean') or 0):.0f}</strong></td>
          <td>{(q.get('thinking_tokens_mean') or 0):.0f}</td>
          <td>{(q.get('answer_tokens_mean') or 0):.0f}</td>
          <td>{q.get('answered', 0)}/{q.get('n', 0)}</td>
        </tr>""")
    full_table_rows = "\n".join(full_rows)
    
    best = _pick_overall(summary)
    rec_html = ""
    if best:
        m_id = best["model"]
        m_cfg = C.MODELS_CONFIG.get(m_id)
        if m_cfg:
            model_path = m_cfg["quants"][best["quant"]]
            dn_args = f"--spec-type draft-mtp --spec-draft-n-max {best['best_draft_n']}" if best.get("best_draft_n") else "--spec-type none"
            rf_arg = f" --reasoning-format {m_cfg['reasoning_format']}" if m_cfg.get("reasoning_format") and m_cfg["reasoning_format"] != "none" else ""
            
            rec_html = f"""
    <div class="recommendation-card">
      <h3 style="margin-top: 0; font-family: var(--font-title); font-size: 1.5rem; color: #fff;">
        🚀 Recommended Hosting Configuration
      </h3>
      <p>Based on comprehensive speed-to-quality optimizations, serve <strong>{best['label']}</strong> on this machine with the following launch command:</p>
      <pre><code>llama-server -m {model_path} \\
  {dn_args} \\
  -c 16384 -ngl 99 -fa on -np 1 --jinja{rf_arg} \\
  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080</code></pre>
    </div>"""
            
    html_content = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8">
  <meta name="viewport" content="width=device-width, initial-scale=1.0">
  <title>Local LLM Benchmark Suite Dashboard</title>
  <link rel="preconnect" href="https://fonts.googleapis.com">
  <link rel="preconnect" href="https://fonts.gstatic.com" crossorigin>
  <link href="https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Plus+Jakarta+Sans:wght@400;500;600;700&display=swap" rel="stylesheet">
  <style>
    :root {{
      --bg: #0b0f19;
      --card-bg: rgba(22, 31, 56, 0.7);
      --card-border: rgba(255, 255, 255, 0.08);
      --text: #e2e8f0;
      --text-muted: #94a3b8;
      --primary: #3b82f6;
      --primary-glow: rgba(59, 130, 246, 0.15);
      --success: #10b981;
      --success-glow: rgba(16, 185, 129, 0.15);
      --accent: #8b5cf6;
      --accent-glow: rgba(139, 92, 246, 0.15);
      --font-title: 'Outfit', sans-serif;
      --font-body: 'Plus Jakarta Sans', sans-serif;
    }}
    body {{
      background-color: var(--bg);
      color: var(--text);
      font-family: var(--font-body);
      margin: 0;
      padding: 0;
      line-height: 1.6;
    }}
    .container {{
      max-width: 1200px;
      margin: 0 auto;
      padding: 40px 20px;
    }}
    header {{
      margin-bottom: 40px;
      border-bottom: 1px solid var(--card-border);
      padding-bottom: 24px;
    }}
    h1 {{
      font-family: var(--font-title);
      font-size: 2.5rem;
      font-weight: 700;
      margin: 0 0 10px 0;
      background: linear-gradient(135deg, #60a5fa, #a78bfa);
      -webkit-background-clip: text;
      -webkit-text-fill-color: transparent;
    }}
    .meta-subtitle {{
      font-size: 0.95rem;
      color: var(--text-muted);
      margin: 0;
    }}
    .grid-tldr {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(350px, 1fr));
      gap: 20px;
      margin-bottom: 40px;
    }}
    .card {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 24px;
      backdrop-filter: blur(12px);
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
      transition: transform 0.2s, box-shadow 0.2s;
    }}
    .card:hover {{
      transform: translateY(-2px);
      box-shadow: 0 8px 30px rgba(0, 0, 0, 0.4);
    }}
    .card-title {{
      font-family: var(--font-title);
      font-size: 1.25rem;
      font-weight: 600;
      margin: 0 0 15px 0;
      color: #fff;
      display: flex;
      align-items: center;
    }}
    .card-stat {{
      font-size: 2.2rem;
      font-weight: 700;
      color: #fff;
      margin-bottom: 10px;
      font-family: var(--font-title);
    }}
    .card-detail {{
      font-size: 0.9rem;
      color: var(--text-muted);
      line-height: 1.5;
    }}
    .badge {{
      display: inline-block;
      padding: 4px 8px;
      border-radius: 6px;
      font-size: 0.75rem;
      font-weight: 600;
      text-transform: uppercase;
      margin-right: 8px;
    }}
    .badge-primary {{
      background: var(--primary-glow);
      color: #60a5fa;
      border: 1px solid rgba(59, 130, 246, 0.3);
    }}
    .badge-success {{
      background: var(--success-glow);
      color: #34d399;
      border: 1px solid rgba(16, 185, 129, 0.3);
    }}
    .badge-accent {{
      background: var(--accent-glow);
      color: #c084fc;
      border: 1px solid rgba(139, 92, 246, 0.3);
    }}
    .section-title {{
      font-family: var(--font-title);
      font-size: 1.75rem;
      font-weight: 600;
      margin: 40px 0 20px 0;
      color: #fff;
      border-left: 4px solid var(--primary);
      padding-left: 12px;
    }}
    .grid-charts {{
      display: grid;
      grid-template-columns: repeat(auto-fill, minmax(500px, 1fr));
      gap: 24px;
      margin-bottom: 40px;
    }}
    @media (max-width: 600px) {{
      .grid-charts {{
        grid-template-columns: 1fr;
      }}
    }}
    .chart-box {{
      background: var(--card-bg);
      border: 1px solid var(--card-border);
      border-radius: 16px;
      padding: 20px;
      text-align: center;
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }}
    .chart-box img {{
      max-width: 100%;
      height: auto;
      border-radius: 8px;
    }}
    .chart-title {{
      font-size: 0.95rem;
      color: var(--text-muted);
      margin-top: 15px;
      font-weight: 500;
    }}
    .table-container {{
      width: 100%;
      overflow-x: auto;
      margin-bottom: 40px;
      border-radius: 12px;
      border: 1px solid var(--card-border);
      box-shadow: 0 4px 30px rgba(0, 0, 0, 0.2);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      background: var(--card-bg);
    }}
    th, td {{
      padding: 14px 20px;
      text-align: left;
    }}
    th {{
      background-color: rgba(255, 255, 255, 0.03);
      color: #fff;
      font-weight: 600;
      border-bottom: 1px solid var(--card-border);
      font-family: var(--font-title);
      font-size: 0.95rem;
    }}
    td {{
      border-bottom: 1px solid rgba(255, 255, 255, 0.04);
      font-size: 0.9rem;
    }}
    tr:last-child td {{
      border-bottom: none;
    }}
    tr:hover {{
      background-color: rgba(255, 255, 255, 0.02);
    }}
    .recommendation-card {{
      background: linear-gradient(135deg, rgba(59, 130, 246, 0.08) 0%, rgba(139, 92, 246, 0.08) 100%);
      border: 1px solid rgba(59, 130, 246, 0.2);
      border-radius: 16px;
      padding: 30px;
      margin-top: 40px;
      box-shadow: 0 4px 30px rgba(59, 130, 246, 0.05);
    }}
    pre {{
      background: #04060a;
      border: 1px solid rgba(255, 255, 255, 0.05);
      border-radius: 12px;
      padding: 20px;
      overflow-x: auto;
      color: #34d399;
      font-family: 'Courier New', Courier, monospace;
      font-size: 0.9rem;
      margin: 20px 0 0 0;
    }}
    code {{
      font-family: inherit;
    }}
  </style>
</head>
<body>
  <div class="container">
    <header>
      <h1>Local LLM Benchmark Suite Dashboard</h1>
      <p class="meta-subtitle">
        Speed sweep: {m['n_speed']} runs · full-length: {m['n_full']} runs · 
        temp {m['temp']}, seed {m['seed']}, ctx {m['ctx']}
      </p>
    </header>

    <div class="section-title">Performance Summary</div>
    <div class="grid-tldr">
      {tldr_html}
    </div>

    <div class="section-title">Visual Analytics</div>
    <div class="grid-charts">
      {charts_html}
    </div>

    <div class="section-title">Speed Sweep Metrics</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Model & Quant</th>
            <th>Config</th>
            <th>Speed (tok/s)</th>
            <th>Margin of Error</th>
            <th>MTP Accept %</th>
            <th>Prefill Speed</th>
            <th>TTFT</th>
          </tr>
        </thead>
        <tbody>
          {speed_table_rows}
        </tbody>
      </table>
    </div>

    <div class="section-title">Full-length Generation Capacity</div>
    <div class="table-container">
      <table>
        <thead>
          <tr>
            <th>Model & Quant</th>
            <th>Total Tokens</th>
            <th>Thinking Tokens</th>
            <th>Answer Tokens</th>
            <th>Completions</th>
          </tr>
        </thead>
        <tbody>
          {full_table_rows}
        </tbody>
      </table>
    </div>

    <div class="section-title">MTP Output Determinism Probe</div>
    <div class="card" style="margin-bottom: 40px;">
      <div class="card-detail">
        <p style="margin: 0; font-size: 1rem;">
          🔍 Out of compared runs, <strong>{summary['determinism']['mtp_vs_off_diverged']}/{summary['determinism']['mtp_vs_off_compared']}</strong> speculative decoding runs diverged from the MTP-off baseline under a fixed seed.
        </p>
      </div>
    </div>

    {rec_html}
  </div>
</body>
</html>"""
    
    with open(os.path.join(C.REPO, "REPORT.html"), "w") as f:
        f.write(html_content)
    print("wrote REPORT.html")


def _pick_overall(summary):
    pq = [q for q in summary["per_quant"] if q.get("best_tok_s")]
    if not pq:
        return None
    graded = [q for q in pq if q.get("quality_overall") is not None]
    if graded:
        return max(graded, key=lambda q: (round(q["quality_overall"], 1), q["best_tok_s"]))
    return max(pq, key=lambda q: q["best_tok_s"])


# ------------------------------------------------------------------------------ main
def main():
    os.makedirs(C.CHARTS, exist_ok=True)
    recs = load_all()

    # Runtime tags in results.jsonl, and why the filters below are explicit:
    #   None / "gguf"  -> llama.cpp. Older records predate the tag, hence the `in`.
    #   "mlx_vlm"      -> the current in-process mlx_vlm harness.
    #   "mlx"          -> the LEGACY mlx_lm.server harness. Deliberately excluded from
    #                     both arms: those runs had no speculative decoding, a corrupted
    #                     thinking/answer split, TTFT-derived prompt speed, and (for the
    #                     27B) came from a different repo on an older harness. They are
    #                     kept in results.jsonl for provenance but must not be reported.
    def _rt(r):
        return r.get("runtime")

    def is_gguf(r):
        return _rt(r) in (None, "gguf")

    def is_mlx(r):
        return _rt(r) == "mlx_vlm"

    legacy = sum(1 for r in recs if _rt(r) == "mlx")
    if legacy:
        print(f"note: ignoring {legacy} legacy mlx_lm.server records (superseded)")

    # speed sweep over all GGUF runs
    speed = []
    for r in recs:
        if r.get("phase") == "speed" and "predicted_per_second" in r and is_gguf(r):
            model = r.get("model", "qwen3.6-27b")
            valid_drafts = [0, 1, 2, 3, 4]
            if model in C.MODELS_CONFIG:
                valid_drafts = C.MODELS_CONFIG[model]["draft_ns"]
            if r.get("draft_n") in valid_drafts:
                speed.append(r)

    # full sweep over GGUF runs
    full = [r for r in recs if r.get("phase") == "full" and r.get("thinking_tokens") is not None
            and is_gguf(r)]

    # MLX records (current harness only)
    mlx_speed = [r for r in recs if is_mlx(r) and r.get("phase") == "speed"
                 and r.get("predicted_per_second")]
    mlx_full = [r for r in recs if is_mlx(r) and r.get("phase") == "full"
                and r.get("thinking_tokens") is not None]

    if not speed:
        print("no speed records yet")
        return
    scores = load_scores()
    summary = build_summary(speed, full, scores, mlx_speed, mlx_full)
    chart_tok_s(summary); chart_speedup(summary); chart_acceptance(summary)
    chart_prompt_speed(summary); chart_ttft(summary)
    if full:
        chart_tokens(summary)
    has_quality = chart_quality(summary)
    chart_quality_speed(summary)
    if mlx_speed:
        chart_runtime_compare(summary)
    write_report(summary, has_quality, bool(mlx_speed))
    write_html_report(summary, has_quality, bool(mlx_speed))
    print(f"done: {len(speed)} speed + {len(full)} full + {len(mlx_speed)} mlx records")


if __name__ == "__main__":
    main()
