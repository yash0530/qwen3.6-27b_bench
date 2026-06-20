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

QUANTS = C.QUANT_ORDER
QUANT_LABEL = {"q5": "Q5_K_XL", "q6": "Q6_K_XL", "q8": "Q8_0", "mlx8": "MLX-8bit"}
COLORS = {"q5": "#2563eb", "q6": "#16a34a", "q8": "#dc2626", "mlx8": "#9333ea"}
MLX_QUANTS = ["mlx8"]  # MLX runtime models (no MTP sweep; single config)


# ------------------------------------------------------------------- load/group
def load_all():
    if os.path.exists(C.RESULTS_JSON):
        with open(C.RESULTS_JSON) as f:
            recs = json.load(f)
    else:
        recs = []
        with open(C.RESULTS_JSONL) as f:
            for ln in f:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    return [r for r in recs if r.get("error") is None]


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
            out.setdefault(item["quant"], {})[item["question_id"]] = item
    return out


# ----------------------------------------------------------------------- aggregate
def build_summary(speed, full, scores, mlx_speed=None, mlx_full=None):
    def sgroup(quant, dn):
        return [r for r in speed if r["quant"] == quant and r["draft_n"] == dn
                and "predicted_per_second" in r]

    per_cell = []
    for quant in QUANTS:
        for dn in C.DRAFT_NS:
            g = sgroup(quant, dn)
            if not g:
                continue
            per_cell.append({
                "quant": quant, "draft_n": dn, "config": "off" if dn == 0 else f"mtp{dn}",
                "n_runs": len(g),
                "tok_s_mean": fmean([r.get("predicted_per_second") for r in g]),
                "tok_s_std": fstd([r.get("predicted_per_second") for r in g]),
                "prompt_tok_s_mean": fmean([r.get("prompt_per_second") for r in g]),
                "ttft_ms_mean": fmean([r.get("ttft_ms") for r in g]),
                "acceptance_mean": fmean([r.get("acceptance_rate") for r in g]),
            })

    # full-length token totals per quant
    tokens = {}
    for quant in QUANTS:
        fg = [r for r in full if r["quant"] == quant]
        if fg:
            tokens[quant] = {
                "thinking_tokens_mean": fmean([r.get("thinking_tokens") for r in fg]),
                "answer_tokens_mean": fmean([r.get("answer_tokens") for r in fg]),
                "total_tokens_mean": fmean([r.get("predicted_n") for r in fg]),
                "answered": sum(1 for r in fg if (r.get("answer_tokens") or 0) > 0),
                "n": len(fg),
            }

    per_quant = []
    for quant in QUANTS:
        cells = [c for c in per_cell if c["quant"] == quant]
        if not cells:
            continue
        off = next((c for c in cells if c["draft_n"] == 0), None)
        mtp = [c for c in cells if c["draft_n"] > 0 and c["tok_s_mean"]]
        best = max(mtp, key=lambda c: c["tok_s_mean"]) if mtp else None
        off_tok = off["tok_s_mean"] if off else None
        q_scores = scores.get(quant, {})
        overall = fmean([s.get("overall") for s in q_scores.values()]) if q_scores else None
        per_quant.append({
            "quant": quant, "label": QUANT_LABEL[quant],
            "off_tok_s": off_tok,
            "best_draft_n": best["draft_n"] if best else None,
            "best_tok_s": best["tok_s_mean"] if best else None,
            "speedup_vs_off": (best["tok_s_mean"] / off_tok) if best and off_tok else None,
            "best_acceptance": best["acceptance_mean"] if best else None,
            "quality_overall": overall,
            **(tokens.get(quant, {})),
        })

    # MLX models: single config (no MTP). tok/s from speed phase; tokens from full phase.
    for mq in (MLX_QUANTS if mlx_speed else []):
        sg = [r for r in mlx_speed if r["quant"] == mq]
        if not sg:
            continue
        fg = [r for r in (mlx_full or []) if r["quant"] == mq]
        tok_s = fmean([r.get("predicted_per_second") for r in sg])
        q_scores = scores.get(mq, {})
        overall = fmean([s.get("overall") for s in q_scores.values()]) if q_scores else None
        per_quant.append({
            "quant": mq, "label": QUANT_LABEL.get(mq, mq), "runtime": "mlx",
            "off_tok_s": tok_s, "best_draft_n": None, "best_tok_s": tok_s,
            "speedup_vs_off": None, "best_acceptance": None, "quality_overall": overall,
            "thinking_tokens_mean": fmean([r.get("thinking_tokens") for r in fg]),
            "answer_tokens_mean": fmean([r.get("answer_tokens") for r in fg]),
            "total_tokens_mean": fmean([r.get("predicted_n") for r in fg]),
            "ttft_ms_mean": fmean([r.get("ttft_ms") for r in sg]),
            "prompt_tok_s_mean": fmean([r.get("prompt_per_second") for r in sg]),
            "n": len(fg), "answered": sum(1 for r in fg if (r.get("answer_tokens") or 0) > 0),
        })

    summary = {
        "meta": {
            "quants": QUANTS, "draft_ns": C.DRAFT_NS, "seed": C.SEED, "temp": C.TEMP,
            "top_p": C.TOP_P, "top_k": C.TOP_K, "speed_cap": C.SPEED_N_PREDICT,
            "full_cap": C.FULL_N_PREDICT, "ctx": C.CTX,
            "n_speed": len(speed), "n_full": len(full),
        },
        "per_quant_config": per_cell,
        "per_quant": per_quant,
        "determinism": determinism_check(speed),
    }
    with open(C.SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def determinism_check(speed):
    """Within the speed phase: how often does an MTP run differ from its off baseline?"""
    base = {}
    for r in speed:
        if r["draft_n"] == 0:
            base[(r["quant"], r["question_id"])] = r.get("output_sha256")
    total = diff = 0
    for r in speed:
        if r["draft_n"] > 0:
            b = base.get((r["quant"], r["question_id"]))
            if b is not None and r.get("output_sha256") is not None:
                total += 1
                if r["output_sha256"] != b:
                    diff += 1
    return {"mtp_vs_off_compared": total, "mtp_vs_off_diverged": diff}


# --------------------------------------------------------------------------- charts
def _xlabels():
    return ["off" if d == 0 else str(d) for d in C.DRAFT_NS]


def _cell(summary, quant, dn):
    return next((c for c in summary["per_quant_config"]
                 if c["quant"] == quant and c["draft_n"] == dn), None)


def chart_tok_s(summary):
    plt.figure(figsize=(9, 5.5))
    for quant in QUANTS:
        xs, ys, es = [], [], []
        for i, dn in enumerate(C.DRAFT_NS):
            c = _cell(summary, quant, dn)
            if c and c["tok_s_mean"]:
                xs.append(i); ys.append(c["tok_s_mean"]); es.append(c["tok_s_std"])
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                         color=COLORS[quant], label=QUANT_LABEL[quant])
    plt.xticks(range(len(C.DRAFT_NS)), _xlabels())
    plt.xlabel("MTP draft tokens (--spec-draft-n-max)")
    plt.ylabel("Decode speed (tokens/sec)")
    plt.title("Decode throughput vs MTP draft depth (mean over 5 questions)")
    plt.grid(True, alpha=0.3); plt.legend()
    _save("01_decode_tok_s.png")


def chart_speedup(summary):
    plt.figure(figsize=(9, 5.5))
    for quant in QUANTS:
        off = _cell(summary, quant, 0)
        if not off or not off["tok_s_mean"]:
            continue
        xs, ys = [], []
        for i, dn in enumerate(C.DRAFT_NS):
            if dn == 0:
                continue
            c = _cell(summary, quant, dn)
            if c and c["tok_s_mean"]:
                xs.append(i); ys.append(c["tok_s_mean"] / off["tok_s_mean"])
        if xs:
            plt.plot(xs, ys, marker="o", color=COLORS[quant], label=QUANT_LABEL[quant])
    plt.axhline(1.0, color="gray", ls="--", alpha=0.6, label="MTP off")
    plt.xticks(range(len(C.DRAFT_NS)), _xlabels())
    plt.xlabel("MTP draft tokens"); plt.ylabel("Speedup vs MTP off (x)")
    plt.title("MTP speedup over baseline"); plt.grid(True, alpha=0.3); plt.legend()
    _save("02_speedup.png")


def chart_acceptance(summary):
    plt.figure(figsize=(9, 5.5))
    for quant in QUANTS:
        xs, ys = [], []
        for i, dn in enumerate(C.DRAFT_NS):
            if dn == 0:
                continue
            c = _cell(summary, quant, dn)
            if c and c["acceptance_mean"] is not None:
                xs.append(i); ys.append(c["acceptance_mean"] * 100)
        if xs:
            plt.plot(xs, ys, marker="o", color=COLORS[quant], label=QUANT_LABEL[quant])
    plt.xticks(range(len(C.DRAFT_NS)), _xlabels())
    plt.xlabel("MTP draft tokens"); plt.ylabel("Draft acceptance (%)")
    plt.title("MTP draft acceptance rate"); plt.grid(True, alpha=0.3); plt.legend()
    _save("03_acceptance.png")


def chart_prompt_speed(summary):
    plt.figure(figsize=(7, 5))
    vals = [fmean([c["prompt_tok_s_mean"] for c in summary["per_quant_config"]
                   if c["quant"] == q and c["prompt_tok_s_mean"]]) or 0 for q in QUANTS]
    plt.bar([QUANT_LABEL[q] for q in QUANTS], vals, color=[COLORS[q] for q in QUANTS])
    plt.ylabel("Prompt processing (tokens/sec)")
    plt.title("Prompt processing speed by quant")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.0f}", ha="center", va="bottom")
    _save("04_prompt_speed.png")


def chart_ttft(summary):
    plt.figure(figsize=(9, 5.5))
    for quant in QUANTS:
        xs, ys = [], []
        for i, dn in enumerate(C.DRAFT_NS):
            c = _cell(summary, quant, dn)
            if c and c["ttft_ms_mean"]:
                xs.append(i); ys.append(c["ttft_ms_mean"])
        if xs:
            plt.plot(xs, ys, marker="o", color=COLORS[quant], label=QUANT_LABEL[quant])
    plt.xticks(range(len(C.DRAFT_NS)), _xlabels())
    plt.xlabel("MTP draft tokens"); plt.ylabel("Time to first token (ms)")
    plt.title("Latency to first token"); plt.grid(True, alpha=0.3); plt.legend()
    _save("05_ttft.png")


def chart_tokens(summary):
    pq = {q["quant"]: q for q in summary["per_quant"]}
    x = np.arange(len(QUANTS)); w = 0.6
    think = [(pq.get(q, {}).get("thinking_tokens_mean") or 0) for q in QUANTS]
    ans = [(pq.get(q, {}).get("answer_tokens_mean") or 0) for q in QUANTS]
    plt.figure(figsize=(8, 5.5))
    plt.bar(x, think, w, label="thinking tokens", color="#9333ea")
    plt.bar(x, ans, w, bottom=think, label="answer tokens", color="#f59e0b")
    plt.xticks(x, [QUANT_LABEL[q] for q in QUANTS])
    plt.ylabel("Mean tokens per answer (full 8192 cap)")
    plt.title("Thinking vs answer token volume by quant")
    plt.legend()
    _save("06_tokens.png")


def chart_quality(summary):
    pq = [q for q in summary["per_quant"] if q.get("quality_overall") is not None]
    if not pq:
        return False
    plt.figure(figsize=(7, 5))
    plt.bar([QUANT_LABEL[q["quant"]] for q in pq], [q["quality_overall"] for q in pq],
            color=[COLORS[q["quant"]] for q in pq])
    plt.ylim(0, 10); plt.ylabel("Judge quality (1-10)")
    plt.title("Answer quality by quant (Claude/Opus judge)")
    for i, q in enumerate(pq):
        plt.text(i, q["quality_overall"], f"{q['quality_overall']:.1f}", ha="center", va="bottom")
    _save("07_quality.png")
    return True


def chart_quality_speed(summary):
    pq = [q for q in summary["per_quant"]
          if q.get("quality_overall") is not None and q.get("best_tok_s")]
    if not pq:
        return False
    plt.figure(figsize=(7.5, 5.5))
    for q in pq:
        plt.scatter(q["best_tok_s"], q["quality_overall"], s=160,
                    color=COLORS.get(q["quant"], "#666"), zorder=3)
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
    """Best decode tok/s across all models: GGUF (best-MTP) vs MLX (no MTP)."""
    pq = [q for q in summary["per_quant"] if q.get("best_tok_s")]
    if not pq:
        return
    plt.figure(figsize=(8, 5))
    labels, vals, cols = [], [], []
    for q in pq:
        tag = f"\n(n={q['best_draft_n']})" if q.get("best_draft_n") is not None else "\n(no MTP)"
        labels.append(q["label"] + tag)
        vals.append(q["best_tok_s"])
        cols.append(COLORS.get(q["quant"], "#666"))
    plt.bar(labels, vals, color=cols)
    plt.ylabel("Best decode speed (tokens/sec)")
    plt.title("Runtime comparison: GGUF+MTP vs MLX (no MTP)")
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
    L.append("# Qwen3.6-27B MTP benchmark — Q5 / Q6 / Q8 on Apple M5 Pro (64 GB)\n")
    L.append(f"_Speed sweep: {m['n_speed']} runs (cap {m['speed_cap']} tok) · "
             f"full-length: {m['n_full']} runs (cap {m['full_cap']} tok) · "
             f"temp {m['temp']}, seed {m['seed']}, ctx {m['ctx']}._\n")

    L.append("## TL;DR\n")
    for q in summary["per_quant"]:
        qual = f"{q['quality_overall']:.1f}/10" if q.get("quality_overall") is not None else "ungraded"
        if q.get("runtime") == "mlx":
            L.append(f"- **{q['label']}** (MLX, no MTP) — **{(q['best_tok_s'] or 0):.1f} tok/s**, "
                     f"quality {qual}, ~{(q.get('total_tokens_mean') or 0):.0f} tok/answer.")
        else:
            spd = f"{q['speedup_vs_off']:.2f}x" if q.get("speedup_vs_off") else "n/a"
            L.append(f"- **{q['label']}** — peak **{(q['best_tok_s'] or 0):.1f} tok/s** at "
                     f"draft-n={q['best_draft_n']} ({spd} vs off, "
                     f"{(q['best_acceptance'] or 0)*100:.0f}% accept); quality {qual}; "
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

    L.append("## Speed sweep (mean over 5 questions)\n")
    L.append("| quant | draft-n | tok/s | ±sd | accept % | prompt tok/s | TTFT ms |")
    L.append("|---|---|---|---|---|---|---|")
    for c in summary["per_quant_config"]:
        acc = f"{c['acceptance_mean']*100:.0f}" if c["acceptance_mean"] is not None else "-"
        L.append(f"| {QUANT_LABEL[c['quant']]} | {c['config']} | {(c['tok_s_mean'] or 0):.1f} | "
                 f"{(c['tok_s_std'] or 0):.1f} | {acc} | {(c['prompt_tok_s_mean'] or 0):.0f} | "
                 f"{(c['ttft_ms_mean'] or 0):.0f} |")
    L.append("")

    L.append("## Full-length output (8192 cap, off config)\n")
    L.append("| quant | total tok | thinking tok | answer tok | answers completed |")
    L.append("|---|---|---|---|---|")
    for q in summary["per_quant"]:
        L.append(f"| {q['label']} | {(q.get('total_tokens_mean') or 0):.0f} | "
                 f"{(q.get('thinking_tokens_mean') or 0):.0f} | "
                 f"{(q.get('answer_tokens_mean') or 0):.0f} | "
                 f"{q.get('answered', 0)}/{q.get('n', 0)} |")
    L.append("")

    d = summary["determinism"]
    L.append("## Output determinism (MTP correctness probe, #23302)\n")
    L.append(f"- {d['mtp_vs_off_diverged']}/{d['mtp_vs_off_compared']} MTP runs produced a "
             f"different output than their MTP-off baseline (fixed seed). 0 = MTP is "
             f"output-preserving on this build; >0 flags the known determinism bug.\n")

    best = _pick_overall(summary)
    if best:
        L.append("## Hosting recommendation\n")
        L.append(f"On this machine, serve **{best['label']}** with "
                 f"`--spec-draft-n-max {best['best_draft_n']}` "
                 f"(~{(best['best_tok_s'] or 0):.0f} tok/s, "
                 f"{(best.get('quality_overall') or 0):.1f}/10 quality). Launch line:\n")
        L.append("```bash")
        L.append(f"llama-server -m {C.MODELS[best['quant']]} \\")
        L.append(f"  --spec-type draft-mtp --spec-draft-n-max {best['best_draft_n']} \\")
        L.append("  -c 16384 -ngl 99 -fa on -np 1 --jinja --reasoning-format deepseek \\")
        L.append("  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080")
        L.append("```\n")
        L.append("Wire it into the research engine as an `openai_compat` provider profile "
                 "(`config/providers.ts`) pointing at `http://127.0.0.1:8080/v1`; the existing "
                 "`lib/analyst/` adapter then drives it as the brain.\n")

    with open(os.path.join(C.REPO, "REPORT.md"), "w") as f:
        f.write("\n".join(L))
    print("wrote REPORT.md")


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
    # GGUF (llama.cpp) records: respect the DRAFT_NS cap (n<=4) and exclude MLX.
    speed = [r for r in recs if r.get("phase") == "speed" and "predicted_per_second" in r
             and r.get("draft_n") in C.DRAFT_NS and r.get("runtime") != "mlx"]
    full = [r for r in recs if r.get("phase") == "full" and r.get("thinking_tokens") is not None
            and r.get("runtime") != "mlx"]
    # MLX records (separate runtime, no MTP).
    mlx_speed = [r for r in recs if r.get("runtime") == "mlx" and r.get("phase") == "speed"
                 and r.get("predicted_per_second")]
    mlx_full = [r for r in recs if r.get("runtime") == "mlx" and r.get("phase") == "full"
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
    print(f"done: {len(speed)} speed + {len(full)} full + {len(mlx_speed)} mlx records")


if __name__ == "__main__":
    main()
