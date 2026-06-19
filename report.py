#!/usr/bin/env python3
"""Aggregate benchmark results, render charts, and write REPORT.md + summary.json.

Reads results/results.json (or results.jsonl). If judge scores are present (either
merged onto records as `judge`, or in results/judging/scores.json) a quality axis is
added. Safe to re-run any time (e.g. after judging).
"""
import json
import os
import statistics as stats

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import config as C
from questions import QUESTIONS

QUANTS = C.QUANT_ORDER
QUANT_LABEL = {"q5": "Q5_K_XL", "q6": "Q6_K_XL", "q8": "Q8_0"}
COLORS = {"q5": "#2563eb", "q6": "#16a34a", "q8": "#dc2626"}


# ------------------------------------------------------------------- load/group
def load_records():
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
    return [r for r in recs if r.get("error") is None and "predicted_per_second" in r]


def fmean(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return stats.mean(xs) if xs else None


def fstd(xs):
    xs = [x for x in xs if isinstance(x, (int, float))]
    return stats.pstdev(xs) if len(xs) > 1 else 0.0


def group(recs, quant, dn):
    return [r for r in recs if r["quant"] == quant and r["draft_n"] == dn]


def load_scores():
    """quant -> question_id -> {scores..., overall, rationale}. From merged judge field or scores.json."""
    out = {}
    sp = os.path.join(C.JUDGING, "scores.json")
    if os.path.exists(sp):
        with open(sp) as f:
            data = json.load(f)
        for item in data.get("scores", []):
            out.setdefault(item["quant"], {})[item["question_id"]] = item
    return out


# ----------------------------------------------------------------------- aggregate
def build_summary(recs, scores):
    per_cell = []
    for quant in QUANTS:
        for dn in C.DRAFT_NS:
            g = group(recs, quant, dn)
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
                "thinking_tokens_mean": fmean([r.get("thinking_tokens") for r in g]),
                "answer_tokens_mean": fmean([r.get("answer_tokens") for r in g]),
                "predicted_n_mean": fmean([r.get("predicted_n") for r in g]),
            })

    per_quant = []
    for quant in QUANTS:
        cells = [c for c in per_cell if c["quant"] == quant]
        if not cells:
            continue
        off = next((c for c in cells if c["draft_n"] == 0), None)
        mtp_cells = [c for c in cells if c["draft_n"] > 0 and c["tok_s_mean"]]
        best = max(mtp_cells, key=lambda c: c["tok_s_mean"]) if mtp_cells else None
        off_tok = off["tok_s_mean"] if off else None
        q_scores = scores.get(quant, {})
        overall = fmean([s.get("overall") for s in q_scores.values()]) if q_scores else None
        per_quant.append({
            "quant": quant,
            "label": QUANT_LABEL[quant],
            "off_tok_s": off_tok,
            "best_draft_n": best["draft_n"] if best else None,
            "best_tok_s": best["tok_s_mean"] if best else None,
            "speedup_vs_off": (best["tok_s_mean"] / off_tok) if best and off_tok else None,
            "best_acceptance": best["acceptance_mean"] if best else None,
            "quality_overall": overall,
        })

    determinism = determinism_check(recs)
    summary = {
        "meta": {
            "passes": C.PASSES, "quants": QUANTS, "draft_ns": C.DRAFT_NS,
            "seed": C.SEED, "temp": C.TEMP, "top_p": C.TOP_P, "top_k": C.TOP_K,
            "n_predict": C.N_PREDICT, "ctx": C.CTX, "n_records": len(recs),
        },
        "per_quant_config": per_cell,
        "per_quant": per_quant,
        "determinism": determinism,
    }
    with open(C.SUMMARY_JSON, "w") as f:
        json.dump(summary, f, indent=2)
    return summary


def determinism_check(recs):
    """How often pass-2 output != pass-1, and draft-n output != off baseline."""
    by_key = {}
    for r in recs:
        by_key.setdefault((r["quant"], r["draft_n"], r["question_id"]), {})[r["pass"]] = r.get("output_sha256")
    cross_pass_total = cross_pass_diff = 0
    for k, byp in by_key.items():
        if 1 in byp and 2 in byp:
            cross_pass_total += 1
            if byp[1] != byp[2]:
                cross_pass_diff += 1

    baseline = {}
    for r in recs:
        if r["draft_n"] == 0 and r["pass"] == 1:
            baseline[(r["quant"], r["question_id"])] = r.get("output_sha256")
    vs_base_total = vs_base_diff = 0
    for r in recs:
        if r["draft_n"] > 0 and r["pass"] == 1:
            b = baseline.get((r["quant"], r["question_id"]))
            if b is not None:
                vs_base_total += 1
                if r.get("output_sha256") != b:
                    vs_base_diff += 1
    return {
        "cross_pass_compared": cross_pass_total,
        "cross_pass_diverged": cross_pass_diff,
        "mtp_vs_off_compared": vs_base_total,
        "mtp_vs_off_diverged": vs_base_diff,
    }


# --------------------------------------------------------------------------- charts
def _xlabels():
    return ["off" if d == 0 else str(d) for d in C.DRAFT_NS]


def chart_tok_s(summary):
    plt.figure(figsize=(9, 5.5))
    for quant in QUANTS:
        xs, ys, es = [], [], []
        for i, dn in enumerate(C.DRAFT_NS):
            c = next((c for c in summary["per_quant_config"]
                      if c["quant"] == quant and c["draft_n"] == dn), None)
            if c and c["tok_s_mean"]:
                xs.append(i); ys.append(c["tok_s_mean"]); es.append(c["tok_s_std"])
        if xs:
            plt.errorbar(xs, ys, yerr=es, marker="o", capsize=3,
                         color=COLORS[quant], label=QUANT_LABEL[quant])
    plt.xticks(range(len(C.DRAFT_NS)), _xlabels())
    plt.xlabel("MTP draft tokens (--spec-draft-n-max)")
    plt.ylabel("Decode speed (tokens/sec)")
    plt.title("Decode throughput vs MTP draft depth (mean +/- sd over 2 passes x 5 Qs)")
    plt.grid(True, alpha=0.3); plt.legend()
    _save("01_decode_tok_s.png")


def chart_speedup(summary):
    plt.figure(figsize=(9, 5.5))
    for quant in QUANTS:
        off = next((c for c in summary["per_quant_config"]
                    if c["quant"] == quant and c["draft_n"] == 0), None)
        if not off or not off["tok_s_mean"]:
            continue
        xs, ys = [], []
        for i, dn in enumerate(C.DRAFT_NS):
            if dn == 0:
                continue
            c = next((c for c in summary["per_quant_config"]
                      if c["quant"] == quant and c["draft_n"] == dn), None)
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
            c = next((c for c in summary["per_quant_config"]
                      if c["quant"] == quant and c["draft_n"] == dn), None)
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
    vals = []
    for quant in QUANTS:
        cells = [c["prompt_tok_s_mean"] for c in summary["per_quant_config"]
                 if c["quant"] == quant and c["prompt_tok_s_mean"]]
        vals.append(fmean(cells) or 0)
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
            c = next((c for c in summary["per_quant_config"]
                      if c["quant"] == quant and c["draft_n"] == dn), None)
            if c and c["ttft_ms_mean"]:
                xs.append(i); ys.append(c["ttft_ms_mean"])
        if xs:
            plt.plot(xs, ys, marker="o", color=COLORS[quant], label=QUANT_LABEL[quant])
    plt.xticks(range(len(C.DRAFT_NS)), _xlabels())
    plt.xlabel("MTP draft tokens"); plt.ylabel("Time to first token (ms)")
    plt.title("Latency to first token"); plt.grid(True, alpha=0.3); plt.legend()
    _save("05_ttft.png")


def chart_tokens(summary):
    plt.figure(figsize=(8, 5.5))
    import numpy as np
    x = np.arange(len(QUANTS)); w = 0.6
    think = [fmean([c["thinking_tokens_mean"] for c in summary["per_quant_config"]
                    if c["quant"] == q]) or 0 for q in QUANTS]
    ans = [fmean([c["answer_tokens_mean"] for c in summary["per_quant_config"]
                  if c["quant"] == q]) or 0 for q in QUANTS]
    plt.bar(x, think, w, label="thinking tokens", color="#9333ea")
    plt.bar(x, ans, w, bottom=think, label="answer tokens", color="#f59e0b")
    plt.xticks(x, [QUANT_LABEL[q] for q in QUANTS])
    plt.ylabel("Mean tokens per answer")
    plt.title("Thinking vs answer token volume by quant")
    plt.legend()
    _save("06_tokens.png")


def chart_quality(summary):
    pq = [q for q in summary["per_quant"] if q.get("quality_overall") is not None]
    if not pq:
        return False
    plt.figure(figsize=(7, 5))
    qs = [q["quant"] for q in pq]
    vals = [q["quality_overall"] for q in pq]
    plt.bar([QUANT_LABEL[q] for q in qs], vals, color=[COLORS[q] for q in qs])
    plt.ylim(0, 10); plt.ylabel("Judge quality (1-10)")
    plt.title("Answer quality by quant (Claude/Opus judge)")
    for i, v in enumerate(vals):
        plt.text(i, v, f"{v:.1f}", ha="center", va="bottom")
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
                    color=COLORS[q["quant"]], zorder=3)
        plt.annotate(f"{QUANT_LABEL[q['quant']]}\n(n={q['best_draft_n']})",
                     (q["best_tok_s"], q["quality_overall"]),
                     textcoords="offset points", xytext=(8, 6))
    plt.xlabel("Best decode speed (tok/s, optimal MTP n)")
    plt.ylabel("Judge quality (1-10)")
    plt.title("Quality vs speed trade-off (top-right is best)")
    plt.grid(True, alpha=0.3)
    _save("08_quality_vs_speed.png")
    return True


def _save(name):
    plt.tight_layout()
    plt.savefig(os.path.join(C.CHARTS, name), dpi=130)
    plt.close()


# --------------------------------------------------------------------------- report
def write_report(summary, has_quality):
    m = summary["meta"]
    lines = []
    lines.append("# Qwen3.6-27B MTP benchmark — Q5 / Q6 / Q8 on Apple M5 Pro (64 GB)\n")
    lines.append(f"_{m['n_records']} successful runs · {len(m['passes'])} passes · "
                 f"draft-n sweep {m['draft_ns']} · temp {m['temp']}, seed {m['seed']}, "
                 f"cap {m['n_predict']} tok · ctx {m['ctx']}._\n")

    # Headline recommendation
    pq = summary["per_quant"]
    lines.append("## TL;DR\n")
    for q in pq:
        spd = f"{q['speedup_vs_off']:.2f}x" if q.get("speedup_vs_off") else "n/a"
        qual = f"{q['quality_overall']:.1f}/10" if q.get("quality_overall") is not None else "ungraded"
        lines.append(f"- **{q['label']}** — best at draft-n={q['best_draft_n']}: "
                     f"**{(q['best_tok_s'] or 0):.1f} tok/s** ({spd} vs MTP-off), "
                     f"acceptance {(q['best_acceptance'] or 0)*100:.0f}%, quality {qual}.")
    lines.append("")

    # Charts
    lines.append("## Charts\n")
    chart_files = ["01_decode_tok_s.png", "02_speedup.png", "03_acceptance.png",
                   "04_prompt_speed.png", "05_ttft.png", "06_tokens.png"]
    if has_quality:
        chart_files += ["07_quality.png", "08_quality_vs_speed.png"]
    for cf in chart_files:
        lines.append(f"![{cf}](charts/{cf})\n")

    # Per-config table
    lines.append("## Per-config detail (mean over 2 passes x 5 questions)\n")
    lines.append("| quant | draft-n | tok/s | ±sd | accept % | prompt tok/s | TTFT ms | think tok | answer tok |")
    lines.append("|---|---|---|---|---|---|---|---|---|")
    for c in summary["per_quant_config"]:
        acc = f"{c['acceptance_mean']*100:.0f}" if c["acceptance_mean"] is not None else "-"
        lines.append(
            f"| {QUANT_LABEL[c['quant']]} | {c['config']} | "
            f"{(c['tok_s_mean'] or 0):.1f} | {(c['tok_s_std'] or 0):.1f} | {acc} | "
            f"{(c['prompt_tok_s_mean'] or 0):.0f} | {(c['ttft_ms_mean'] or 0):.0f} | "
            f"{(c['thinking_tokens_mean'] or 0):.0f} | {(c['answer_tokens_mean'] or 0):.0f} |")
    lines.append("")

    # Determinism
    d = summary["determinism"]
    lines.append("## Output determinism\n")
    lines.append(f"- Cross-pass (fixed seed): {d['cross_pass_diverged']}/{d['cross_pass_compared']} "
                 f"cells differed between pass 1 and pass 2.")
    lines.append(f"- MTP vs off baseline (#23302 probe): {d['mtp_vs_off_diverged']}/{d['mtp_vs_off_compared']} "
                 f"MTP runs produced a different output than their MTP-off baseline.\n")

    # Hosting recommendation
    best = _pick_overall(summary)
    if best:
        lines.append("## Hosting recommendation\n")
        lines.append(f"For this machine, **{best['label']}** at "
                     f"`--spec-draft-n-max {best['best_draft_n']}` is the recommended "
                     f"serving config (best measured quality/speed balance). Launch line:\n")
        lines.append("```bash")
        lines.append(f"llama-server -m {C.MODELS[best['quant']]} \\")
        lines.append(f"  --spec-type draft-mtp --spec-draft-n-max {best['best_draft_n']} \\")
        lines.append("  -c 16384 -ngl 99 -fa on -np 1 --jinja --reasoning-format deepseek \\")
        lines.append("  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080")
        lines.append("```\n")
        lines.append("Wire it into the research engine as an `openai_compat` provider profile "
                     "(`config/providers.ts`) pointing at `http://127.0.0.1:8080/v1`; the engine's "
                     "existing `lib/analyst/` OpenAI-compatible adapter then drives it as the brain.\n")

    with open(os.path.join(C.REPO, "REPORT.md"), "w") as f:
        f.write("\n".join(lines))
    print("wrote REPORT.md")


def _pick_overall(summary):
    pq = [q for q in summary["per_quant"] if q.get("best_tok_s")]
    if not pq:
        return None
    graded = [q for q in pq if q.get("quality_overall") is not None]
    if graded:
        # maximize quality, tie-break on speed
        return max(graded, key=lambda q: (round(q["quality_overall"], 1), q["best_tok_s"]))
    return max(pq, key=lambda q: q["best_tok_s"])


# ------------------------------------------------------------------------------ main
def main():
    os.makedirs(C.CHARTS, exist_ok=True)
    recs = load_records()
    if not recs:
        print("no successful records yet")
        return
    scores = load_scores()
    summary = build_summary(recs, scores)
    chart_tok_s(summary); chart_speedup(summary); chart_acceptance(summary)
    chart_prompt_speed(summary); chart_ttft(summary); chart_tokens(summary)
    has_quality = chart_quality(summary)
    chart_quality_speed(summary)
    write_report(summary, has_quality)
    print(f"done: {len(recs)} records, charts in {C.CHARTS}")


if __name__ == "__main__":
    main()
