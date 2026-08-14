#!/usr/bin/env python3
"""One table: every planned Qwen 3.8 benchmark cell, done or pending.

A long sweep is only legible if you can see what has landed and what has not in the same
place. This reads results.jsonl and the auxiliary jsonl files, joins them against the
planned matrix, and prints one row per candidate with a status for each stage. Cells that
have not run yet say so rather than being blank, so an interrupted run is distinguishable
from a run that produced nothing.

Numbers are means over the 5 benchmark questions. `off` and `mtp` are decode tok/s;
`acc` is MTP acceptance. Emits GitHub-flavoured markdown.

    python3 status_table.py              # markdown to stdout
    python3 status_table.py --html out.html
"""
import argparse
import collections
import json
import os
import statistics as st

import config as C

MODEL = "qwen3.8-27b"
SCREEN_DEPTH = 2      # GGUF draft_n used in stage A
SCREEN_BLOCK = 3      # MLX draft_block_size used in stage A

# Stage A screens at these depths; stage C sweeps the survivors across the full range.
STAGES = [
    ("A", "quant screen (shallow + agent)"),
    ("B", "full-length answers for quality grading"),
    ("C", "MTP depth sweep on survivors + drafter precision"),
    ("D", "deep tier 64k + drift + append-only"),
]


def load():
    out = []
    if not os.path.exists(C.RESULTS_JSONL):
        return out
    with open(C.RESULTS_JSONL) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("model") == MODEL and not r.get("error") and not r.get("smoke"):
                out.append(r)
    return out


def read_aux(name):
    p = os.path.join(C.RESULTS, name)
    if not os.path.exists(p):
        return []
    rows = []
    with open(p) as f:
        for ln in f:
            try:
                rows.append(json.loads(ln))
            except Exception:
                pass
    return rows


def size_of(path):
    try:
        if os.path.isfile(path):
            return os.path.getsize(path) / 1e9
        if os.path.isdir(path):
            return sum(os.path.getsize(os.path.join(path, f))
                       for f in os.listdir(path) if f.endswith(".safetensors")) / 1e9
    except OSError:
        pass
    return None


def candidates():
    """(runtime, quant, label, path) for every planned candidate, in screen order."""
    rows = []
    g = C.MODELS_CONFIG[MODEL]
    for q in g["quant_order"]:
        rows.append(("gguf", q, q, g["quants"][q]))
    m = C.MLX_MODELS_CONFIG[MODEL]
    for q in m["quant_order"]:
        rows.append(("mlx", q, q, m["quants"][q]))
    return rows


def fmt(v, spec=".2f", dash="·"):
    return dash if v is None else format(v, spec)


def build():
    recs = load()
    by = collections.defaultdict(list)
    for r in recs:
        rt = "mlx" if r.get("runtime") == "mlx_vlm" else "gguf"
        depth = r.get("draft_block_size") if rt == "mlx" else r.get("draft_n")
        by[(rt, r.get("quant"), r.get("phase"), r.get("prompt_tier"), depth or 0)].append(r)

    def agg(key, field):
        rs = by.get(key)
        if not rs:
            return None
        vals = [r[field] for r in rs if r.get(field) is not None]
        return st.mean(vals) if vals else None

    lines = []
    lines.append(f"| Runtime | Quant | Size | "
                 f"shallow off | shallow MTP | agent off | agent MTP | acc | "
                 f"deep MTP | Full | Swept |")
    lines.append("|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|:--:|")

    for rt, quant, label, path in candidates():
        d = SCREEN_BLOCK if rt == "mlx" else SCREEN_DEPTH
        gb = size_of(path)
        s_off = agg((rt, quant, "speed", "shallow", 0), "predicted_per_second")
        s_mtp = agg((rt, quant, "speed", "shallow", d), "predicted_per_second")
        a_off = agg((rt, quant, "speed", "agent", 0), "predicted_per_second")
        a_mtp = agg((rt, quant, "speed", "agent", d), "predicted_per_second")
        acc = agg((rt, quant, "speed", "agent", d), "acceptance_rate")
        deep = agg((rt, quant, "speed", "deep", d), "predicted_per_second")
        full = any(k[0] == rt and k[1] == quant and k[2] == "full" for k in by)
        # Stage C leaves rows at depths other than the screening one.
        swept = any(k[0] == rt and k[1] == quant and k[2] == "speed"
                    and k[3] == "agent" and k[4] not in (0, d) for k in by)
        lines.append(
            f"| {'llama.cpp' if rt == 'gguf' else 'MLX'} | `{label}` | "
            f"{fmt(gb, '.1f')} GB | {fmt(s_off)} | {fmt(s_mtp)} | "
            f"{fmt(a_off)} | {fmt(a_mtp)} | {fmt(acc, '.0%') if acc else '·'} | "
            f"{fmt(deep)} | {'yes' if full else '·'} | {'yes' if swept else '·'} |")

    # Auxiliary measurements, same status treatment.
    lines.append("")
    lines.append("| Auxiliary run | Status | Result |")
    lines.append("|---|:--:|---|")
    ao = [r for r in read_aux("append_only.jsonl") if r.get("model") == MODEL]
    if ao:
        r = ao[-1]
        lines.append(f"| Append-only cache continuation (MLX) | done | "
                     f"{r['ttft_cold_s']:.2f}s → {r['ttft_warm_s']:.2f}s "
                     f"({r['speedup']:.0f}x), output "
                     f"{'preserved' if r['output_preserved'] else 'diverges'} |")
    else:
        lines.append("| Append-only cache continuation (MLX) | pending | stage D |")
    dr = [r for r in read_aux("drift.jsonl") if r.get("model") == MODEL]
    if dr:
        r = dr[-1]
        lines.append(f"| Machine drift control | done | "
                     f"decode {r['drift_tok_pct']:+.1f}%, prefill "
                     f"{r['drift_prefill_pct']:+.1f}% |")
    else:
        lines.append("| Machine drift control | pending | stage D |")
    qa = os.path.join(C.RESULTS, "quant_agreement.json")
    lines.append(f"| Quant agreement vs reference | "
                 f"{'done' if os.path.exists(qa) else 'pending'} | after stage B |")
    lines.append("| Quality grading (rubric) | pending | after stage B |")

    total = len(recs)
    lines.append("")
    lines.append(f"*{total} measurement rows recorded for {MODEL}. "
                 f"`·` = not yet run. Decode figures are tok/s, mean of 5 questions; "
                 f"GGUF MTP at n={SCREEN_DEPTH}, MLX at block {SCREEN_BLOCK}.*")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--html", default=None, help="also write a standalone HTML page")
    args = ap.parse_args()
    md = build()
    print(md)
    if args.html:
        with open(args.html, "w") as f:
            f.write(md)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
