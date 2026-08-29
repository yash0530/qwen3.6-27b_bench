#!/usr/bin/env python3
"""Summarize per-request prefill warmth from a llama-server log."""
import re, sys
log = sys.argv[1] if len(sys.argv) > 1 else '/Users/yash/Desktop/Programming/local_llm_bench/results/overnight/llamacpp-server.log'
rows = []
for line in open(log, errors='ignore'):
    m = re.search(r'prompt eval time\s*=\s*([0-9.]+)\s*ms\s*/\s*([0-9]+)\s*tokens.*\(\s*([0-9.]+)\s*tokens per second', line)
    if m:
        ms, n, tps = float(m.group(1)), int(m.group(2)), float(m.group(3))
        rows.append((n, ms/1000, tps))
    elif 'full prompt re-processing' in line or 'erased invalidated' in line:
        rows.append(('REPROCESS', None, None))
print(f"{'prompt_tok':>10} {'prefill_s':>9} {'tok/s':>9}")
cold = warm = 0
for n, s, tps in rows:
    if n == 'REPROCESS':
        print(f"{'FULL-REPROCESS':>14}"); cold += 1; continue
    tag = 'WARM' if tps > 3000 else 'cold'
    if tps > 3000: warm += 1
    else: cold += 1
    print(f"{n:>10} {s:>9.2f} {tps:>9.0f}  {tag}")
print(f"\nwarm={warm} cold={cold}")
