#!/usr/bin/env python3
"""Quick analysis of the benchmark results for reporting."""
import json, statistics as stats

with open("/Users/yash/Desktop/Programming/qwen3.6-27b_bench/results/results.jsonl") as f:
    recs = [json.loads(ln) for ln in f if ln.strip()]

# Filter out errors
recs = [r for r in recs if r.get("error") is None]

print(f"Total valid records: {len(recs)}")
print(f"Phases: {set(r['phase'] for r in recs)}")
print(f"Quants: {sorted(set(r['quant'] for r in recs))}")
print(f"Draft-n values: {sorted(set(r['draft_n'] for r in recs))}")
print(f"Questions: {sorted(set(r['question_id'] for r in recs))}")
print(f"Configs: {sorted(set(r['config'] for r in recs))}")
print()

speed = [r for r in recs if r["phase"] == "speed"]
full = [r for r in recs if r["phase"] == "full"]
print(f"Speed-phase records: {len(speed)}")
print(f"Full-phase records: {len(full)}")
print()

# -- Speed analysis per quant x draft_n --
print("=" * 90)
print(f"{'quant':<6} {'draft_n':<8} {'config':<8} {'tok/s mean':>10} {'tok/s std':>10} {'accept%':>9} {'prompt t/s':>10} {'TTFT ms':>9} {'wall_ms':>10} {'n':>3}")
print("-" * 90)
for quant in ["q5", "q6", "q8"]:
    for dn in sorted(set(r["draft_n"] for r in speed)):
        g = [r for r in speed if r["quant"] == quant and r["draft_n"] == dn]
        if not g:
            continue
        tok_s = [r["predicted_per_second"] for r in g if r.get("predicted_per_second")]
        acc = [r["acceptance_rate"] for r in g if r.get("acceptance_rate") is not None]
        prompt_s = [r["prompt_per_second"] for r in g if r.get("prompt_per_second")]
        ttft = [r["ttft_ms"] for r in g if r.get("ttft_ms")]
        wall = [r["wall_ms"] for r in g if r.get("wall_ms")]
        config = g[0]["config"]
        acc_str = f"{stats.mean(acc)*100:.1f}" if acc else "-"
        print(f"{quant:<6} {dn:<8} {config:<8} {stats.mean(tok_s):>10.2f} {stats.pstdev(tok_s):>10.2f} {acc_str:>9} {stats.mean(prompt_s):>10.1f} {stats.mean(ttft):>9.1f} {stats.mean(wall):>10.1f} {len(g):>3}")
    print()

# -- Best MTP config per quant --
print("=" * 90)
print("BEST MTP CONFIG PER QUANT (highest tok/s)")
print("-" * 90)
for quant in ["q5", "q6", "q8"]:
    off_g = [r for r in speed if r["quant"] == quant and r["draft_n"] == 0]
    off_tok = stats.mean([r["predicted_per_second"] for r in off_g]) if off_g else 0
    
    best_dn, best_tok = 0, off_tok
    for dn in range(1, 9):
        g = [r for r in speed if r["quant"] == quant and r["draft_n"] == dn]
        if not g:
            continue
        avg = stats.mean([r["predicted_per_second"] for r in g])
        if avg > best_tok:
            best_dn, best_tok = dn, avg
    
    speedup = best_tok / off_tok if off_tok else 0
    best_g = [r for r in speed if r["quant"] == quant and r["draft_n"] == best_dn]
    acc = [r["acceptance_rate"] for r in best_g if r.get("acceptance_rate") is not None]
    acc_str = f"{stats.mean(acc)*100:.1f}%" if acc else "-"
    
    print(f"  {quant}: off={off_tok:.2f} tok/s → best at draft_n={best_dn}: {best_tok:.2f} tok/s ({speedup:.2f}x speedup, {acc_str} acceptance)")

print()

# -- Full-phase analysis --
if full:
    print("=" * 90)
    print("FULL-PHASE OUTPUT (complete answers, 12288 token cap)")
    print("-" * 90)
    for quant in ["q5", "q6", "q8"]:
        fg = [r for r in full if r["quant"] == quant]
        if not fg:
            continue
        think = [r.get("thinking_tokens", 0) or 0 for r in fg]
        ans = [r.get("answer_tokens", 0) or 0 for r in fg]
        total = [r.get("predicted_n", 0) or 0 for r in fg]
        answered = sum(1 for r in fg if (r.get("answer_tokens") or 0) > 0)
        print(f"  {quant}: {len(fg)} runs, avg thinking={stats.mean(think):.0f}, avg answer={stats.mean(ans):.0f}, avg total={stats.mean(total):.0f}, completed={answered}/{len(fg)}")

print()

# -- Determinism check --
print("=" * 90)
print("DETERMINISM CHECK (MTP vs off baseline, SHA256 comparison)")
print("-" * 90)
base = {}
for r in speed:
    if r["draft_n"] == 0:
        base[(r["quant"], r["question_id"])] = r.get("output_sha256")

total = diff = 0
diverged_details = []
for r in speed:
    if r["draft_n"] > 0:
        b = base.get((r["quant"], r["question_id"]))
        if b is not None and r.get("output_sha256") is not None:
            total += 1
            if r["output_sha256"] != b:
                diff += 1
                diverged_details.append(f"    {r['quant']}/{r['question_id']}/draft_n={r['draft_n']}")

print(f"  Compared: {total}, Diverged: {diff}")
if diverged_details:
    print("  Diverged runs:")
    for d in diverged_details[:20]:
        print(d)
print()

# -- Per-question speed breakdown --
print("=" * 90)
print("PER-QUESTION BASELINE SPEED (draft_n=0)")
print("-" * 90)
print(f"{'question':<25} {'q5 tok/s':>10} {'q6 tok/s':>10} {'q8 tok/s':>10}")
print("-" * 65)
for qid in sorted(set(r["question_id"] for r in speed)):
    vals = {}
    for quant in ["q5", "q6", "q8"]:
        g = [r for r in speed if r["quant"] == quant and r["draft_n"] == 0 and r["question_id"] == qid]
        vals[quant] = f"{g[0]['predicted_per_second']:.2f}" if g else "-"
    print(f"{qid:<25} {vals['q5']:>10} {vals['q6']:>10} {vals['q8']:>10}")

print()

# -- Acceptance rate curve --
print("=" * 90)
print("MTP ACCEPTANCE RATE BY DRAFT-N (mean across questions)")
print("-" * 90)
print(f"{'draft_n':<10} {'q5':>10} {'q6':>10} {'q8':>10}")
print("-" * 45)
for dn in range(1, 9):
    vals = {}
    for quant in ["q5", "q6", "q8"]:
        g = [r for r in speed if r["quant"] == quant and r["draft_n"] == dn]
        acc = [r["acceptance_rate"] for r in g if r.get("acceptance_rate") is not None]
        vals[quant] = f"{stats.mean(acc)*100:.1f}%" if acc else "-"
    print(f"{dn:<10} {vals['q5']:>10} {vals['q6']:>10} {vals['q8']:>10}")
