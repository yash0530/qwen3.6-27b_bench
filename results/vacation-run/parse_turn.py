#!/usr/bin/env python3
"""Parse one llama-server log slice covering a single Claude Code turn."""
import re, sys

lines = open(sys.argv[1], errors='replace').read().splitlines()
reqs = []


def get(t):
    d = next((r for r in reqs if r['task'] == t), None)
    if d is None:
        d = {'task': t}
        reqs.append(d)
    return d


sel = []
for l in lines:
    m = re.search(r'task (\d+) \| prompt eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second', l)
    if m:
        d = get(int(m.group(1)))
        d['pe_s'] = float(m.group(2)) / 1000
        d['pe_tok'] = int(m.group(3))
        d['pe_tps'] = float(m.group(4))
    m = re.search(r'task (\d+) \|\s+eval time =\s*([\d.]+) ms /\s*(\d+) tokens.*?([\d.]+) tokens per second', l)
    if m:
        d = get(int(m.group(1)))
        d['gen'] = int(m.group(3))
        d['dec'] = float(m.group(4))
    m = re.search(r'task (\d+) \| draft acceptance = ([\d.]+) .*mean len =\s*([\d.]+)', l)
    if m:
        d = get(int(m.group(1)))
        d['draft'] = float(m.group(2))
        d['mlen'] = float(m.group(3))
    m = re.search(r'task (\d+) \| stop processing: n_tokens = (\d+)', l)
    if m:
        get(int(m.group(1)))['ctx'] = int(m.group(2))
    m = re.search(r'selected slot by (\S+).*?f_sim_best = ([\d.]+).*?f_keep = ([\d.]+)', l)
    if m:
        sel.append((m.group(1), float(m.group(2)), float(m.group(3))))

tot_c = tot_p = 0
for i, r in enumerate(reqs):
    pe = r.get('pe_tok', 0)
    cached = max(0, r.get('ctx', 0) - r.get('gen', 0) - pe)
    r['cached'] = cached
    r['prompt'] = cached + pe
    tot_c += cached
    tot_p += r['prompt']
    fk = sel[i][2] if i < len(sel) else float('nan')
    print(f"  req{i+1} task={r['task']} prompt={r['prompt']} cached={cached} "
          f"({100*cached/max(1,r['prompt']):.0f}%) reproc={pe} f_keep={fk:.3f} "
          f"prefill={r.get('pe_tps',0):.1f}t/s ({r.get('pe_s',0):.0f}s) "
          f"decode={r.get('dec',0):.2f}t/s gen={r.get('gen',0)} draft={r.get('draft',0):.3f} ctx_end={r.get('ctx',0)}")
print(f"  SUMMARY reqs={len(reqs)} cached_frac={tot_c/max(1,tot_p):.3f} "
      f"max_ctx={max([r.get('ctx',0) for r in reqs], default=0)} "
      f"prefill_s_total={sum(r.get('pe_s',0) for r in reqs):.0f}")
