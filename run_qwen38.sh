#!/usr/bin/env bash
# Qwen 3.8 27B quant sweep: 9 GGUF quants x 2 uploaders vs 3 MLX targets. ~16-20h.
#
# The question is which quant to actually serve, so the stages are ordered to answer that
# first and refine later: screen every candidate at one MTP setting, judge quality, and
# only then spend hours sweeping speculative depth on the two or three that survive.
# Sweeping MTP across all twelve up front would cost a day to tune quants that lose on
# quality anyway.
#
# Serialization is mandatory. Two engines sharing the GPU and the unified memory bus
# measure contention, not engines — and an earlier 3.6 sweep was I/O-starved into
# uselessness by a concurrent model download. This script refuses to start while a
# download is running.
#
# Every stage is resumable: each harness skips (phase, model, quant, block, kv, tier,
# question) cells already in results.jsonl, so re-running continues rather than repeats.
set -uo pipefail
cd "$(dirname "$0")"

PY=.mlxenv/bin/python
L=results
M=qwen3.8-27b
log() { echo "[$(date +%T)] $*" | tee -a "$L/qwen38.log"; }
stage() { log "STAGE $*"; }

# --- refuse to time anything while the disk is busy --------------------------------
if pgrep -f "download_qwen38.py" > /dev/null; then
  log "FATAL: download_qwen38.py is still running. Timed runs must not share I/O."
  exit 1
fi

log "=== qwen 3.8 27B sweep start ==="
$PY - <<'PY' | tee -a "$L/qwen38.log"
import os, config as C
missing = []
for key, path in C.MODELS_CONFIG["qwen3.8-27b"]["quants"].items():
    if not os.path.isfile(path):
        missing.append(f"gguf/{key}")
for key, path in C.MLX_MODELS_CONFIG["qwen3.8-27b"]["quants"].items():
    if not os.path.isdir(path):
        missing.append(f"mlx/{key}")
print(f"missing candidates: {missing or 'none'}")
PY

# ---------------------------------------------------------------- A. quant screen
# One MTP setting (unsloth's dense recommendation), fp16 KV, shallow + agent depth.
# This is the stage that ranks the quants; everything after it refines a shortlist.
# MLX runs before the remaining GGUF quants: the Q8-class GGUF cells that decide the
# llama.cpp-vs-MLX question are already measured, so the MLX arm is now the highest-value
# unknown, while the Q4 quants only refine a curve whose shape is established.
stage "A1/6  MLX quant screen: 3 targets, block 3, shallow+agent"
$PY -u bench_mlx.py --model "$M" --phase speed \
  --tiers shallow,agent --blocks 0,3 --kv-bits 0 >> "$L/qwen38_mlx.log" 2>&1
log "  exit=$?"

# --draft-ns 0,2 is the whole point of the staging: MTP off plus one representative
# depth. Without it bench.py sweeps the model's full draft_ns on every quant, which is
# five times the work and spends most of it tuning quants that lose on quality anyway.
# The Q5 pair was dropped mid-sweep and deleted; already-measured quants resume-skip.
stage "A2/6  GGUF quant screen: remaining quants, mtp off + n=2, shallow+agent"
for q in q8; do
  log "  quant $q"
  python3 -u bench.py --model "$M" --quant "$q" --phase speed --draft-ns 0,2 \
    --tiers shallow,agent --kv-quant none >> "$L/qwen38_gguf.log" 2>&1
  log "    exit=$?"
done

python3 bench.py --consolidate-only >> "$L/qwen38.log" 2>&1

# ------------------------------------------------------- B. full phase for judging
# Unbounded generations, MTP off, fp16 KV, shallow — the canonical config, matched
# across arms so the judge compares answers rather than serving settings.
stage "B3/6  full-length phase for quality grading (all candidates)"
for q in q8; do
  python3 -u bench.py --model "$M" --quant "$q" --phase full >> "$L/qwen38_gguf.log" 2>&1
done
$PY -u bench_mlx.py --model "$M" --phase full >> "$L/qwen38_mlx.log" 2>&1
log "  exit=$?"

python3 bench.py --consolidate-only >> "$L/qwen38.log" 2>&1
python3 quant_agreement.py >> "$L/qwen38.log" 2>&1

# ------------------------------------------------------------------ C. MTP sweep
# Only on the survivors, chosen from the screen rather than assumed. Depth is where
# speculation earns or loses its keep, so this runs at agent depth.
stage "C4/6  MTP sweep on the top quants (from stage A data)"
TOP_GGUF=$($PY - <<'PY' 2>/dev/null || echo q8
import json, collections, statistics as st
c = collections.defaultdict(list)
for ln in open('results/results.jsonl'):
    try: r = json.loads(ln)
    except Exception: continue
    if (r.get('model') == 'qwen3.8-27b' and r.get('runtime') == 'gguf'
            and not r.get('error') and not r.get('smoke')
            and r.get('prompt_tier') == 'agent' and r.get('predicted_per_second')):
        c[r['quant']].append(r['predicted_per_second'])
print(' '.join(sorted(c, key=lambda k: -st.mean(c[k]))[:3]) if c else 'q8')
PY
)
log "  sweeping MTP on: $TOP_GGUF"
for q in $TOP_GGUF; do
  python3 -u bench.py --model "$M" --quant "$q" --phase speed \
    --tiers agent --kv-quant none >> "$L/qwen38_gguf.log" 2>&1
done

stage "C5/6  MLX block sweep + drafter precision control"
$PY -u bench_mlx.py --model "$M" --phase speed \
  --tiers agent --blocks 0,2,3,4,5 --kv-bits 0 >> "$L/qwen38_mlx.log" 2>&1
# The only 3.8 MLX drafter is third-party and quantized; on 3.6 quantized MTP heads
# were reported to collapse acceptance. Measure the bf16 head against the 8-bit one
# rather than inheriting either claim.
$PY -u bench_mlx.py --model "$M" --phase speed --draft-alt \
  --tiers agent --blocks 3 --kv-bits 0 >> "$L/qwen38_mlx.log" 2>&1
log "  exit=$?"

python3 bench.py --consolidate-only >> "$L/qwen38.log" 2>&1

# ------------------------------------------------------------- D. deep tier (64k)
stage "D6/6  deep tier on the winner, plus controls"
python3 -u drift_check.py >> "$L/qwen38.log" 2>&1
# Deep-tier prefill costs minutes per generation, so this is MTP off plus the one depth
# that actually won at agent depth — chosen from the stage C data rather than assumed.
# A full depth sweep at 64k would spend hours refining a curve already established.
BEST_N=$($PY - <<'PY' 2>/dev/null || echo 2
import json, collections, statistics as st
c = collections.defaultdict(list)
for ln in open('results/results.jsonl'):
    try: r = json.loads(ln)
    except Exception: continue
    if (r.get('model') == 'qwen3.8-27b' and r.get('runtime') == 'gguf'
            and not r.get('error') and not r.get('smoke')
            and r.get('prompt_tier') == 'agent' and (r.get('draft_n') or 0) > 0
            and r.get('predicted_per_second')):
        c[r['draft_n']].append(r['predicted_per_second'])
print(max(c, key=lambda k: st.mean(c[k])) if c else 2)
PY
)
log "  deep-tier draft depth chosen from agent data: n=$BEST_N"
for q in $(echo "$TOP_GGUF" | cut -d' ' -f1-2); do
  python3 -u bench.py --model "$M" --quant "$q" --phase speed --draft-ns "0,$BEST_N" \
    --tiers deep --kv-quant none >> "$L/qwen38_gguf.log" 2>&1
done
$PY -u bench_mlx.py --model "$M" --phase speed \
  --tiers deep --blocks 0,3 --kv-bits 0 >> "$L/qwen38_mlx.log" 2>&1

# Does 3.8 inherit 3.6's warm-cache story? Same hybrid cache, so expected — but the
# append-only result is the one that decides whether MLX is viable, so it is measured.
$PY -u bench_append_only.py --model "$M" >> "$L/qwen38.log" 2>&1
python3 -u drift_check.py >> "$L/qwen38.log" 2>&1

stage "consolidate + validate + report"
python3 bench.py --consolidate-only >> "$L/qwen38.log" 2>&1
python3 validate_parity.py >> "$L/qwen38.log" 2>&1 || log "  !! PARITY FAILED — read before trusting the report"
python3 judge_export.py >> "$L/qwen38.log" 2>&1
python3 report.py >> "$L/qwen38.log" 2>&1

log "=== qwen 3.8 27B sweep complete ==="
