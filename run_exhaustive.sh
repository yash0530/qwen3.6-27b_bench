#!/usr/bin/env bash
# Full benchmark matrix, serialized. Expect ~10-12h.
#
# Ordering is deliberate: stages run most-decision-relevant first, so that whenever this
# is interrupted or inspected, the data that exists is the data that matters. The 27B at
# agent depth is the cell that decides the serving config; deep-tier refinement of a
# block-size sweep we already know the shape of is worth far less per hour.
#
# Serialization is mandatory, not tidiness — two runtimes sharing the GPU and the unified
# memory bus produce numbers that measure contention rather than the engines. An earlier
# run that overlapped with a model download was I/O-starved into uselessness.
#
# Every stage is resumable: each harness skips (phase, model, quant, block, kv, tier,
# question) cells already present in results.jsonl, so re-running resumes rather than
# duplicating.
set -uo pipefail
cd "$(dirname "$0")"

PY=.mlxenv/bin/python
L=results
log() { echo "[$(date +%T)] $*" | tee -a "$L/exhaustive.log"; }
stage() { log "STAGE $*"; }

log "=== exhaustive run start ==="

# ---------------------------------------------------------------- 1. 27B, shallow+agent
stage "1/8  27B MLX  shallow+agent  blocks 0,2,3,4,5  kv fp16,q8"
$PY -u bench_mlx.py --model qwen3.6-27b --phase speed \
  --tiers shallow,agent --blocks 0,2,3,4,5 --kv-bits 0,8 >> "$L/mlx_27b.log" 2>&1
log "  exit=$?"

stage "2/8  27B GGUF shallow+agent  draft 0-4  kv fp16,q8_0"
python3 -u bench.py --model qwen3.6-27b --quant q8 --phase speed \
  --tiers shallow,agent --kv-quant none,q8_0 >> "$L/gguf_27b.log" 2>&1
log "  exit=$?"

# ---------------------------------------------------------------- 2. 35B, shallow+agent
stage "3/8  35B MLX  shallow+agent  blocks 0,2,3,4,5  kv fp16,q8"
$PY -u bench_mlx.py --model qwen3.6-35b-a3b --phase speed \
  --tiers shallow,agent --blocks 0,2,3,4,5 --kv-bits 0,8 >> "$L/mlx_35b.log" 2>&1
log "  exit=$?"

stage "4/8  35B GGUF shallow+agent  draft 0-4  kv fp16,q8_0"
python3 -u bench.py --model qwen3.6-35b-a3b --quant q8 --phase speed \
  --tiers shallow,agent --kv-quant none,q8_0 >> "$L/gguf_35b.log" 2>&1
log "  exit=$?"

python3 bench.py --consolidate-only >> "$L/exhaustive.log" 2>&1

# ------------------------------------------------------------------ 3. deep tier (64k)
# Targeted, not exhaustive: only MTP-off plus the block size that actually won at agent
# depth, chosen from the data rather than assumed. A full block sweep at 64k would cost
# ~10h to refine a curve whose shape is already established.
best_block() {  # $1=model  -> best mlx draft_block_size at agent tier, fallback 3
  $PY - "$1" <<'PY' 2>/dev/null || echo 3
import json,sys,collections,statistics as st
m=sys.argv[1]
try: recs=[json.loads(l) for l in open('results/results.jsonl')]
except Exception: print(3); raise SystemExit
c=collections.defaultdict(list)
for r in recs:
    if (r.get('runtime')=='mlx_vlm' and r.get('model')==m and not r.get('error')
            and r.get('prompt_tier')=='agent' and (r.get('draft_block_size') or 0)>0
            and r.get('predicted_per_second')):
        c[r['draft_block_size']].append(r['predicted_per_second'])
print(max(c, key=lambda k: st.mean(c[k])) if c else 3)
PY
}

B27=$(best_block qwen3.6-27b); B35=$(best_block qwen3.6-35b-a3b)
log "deep-tier block sizes chosen from agent data: 27B=$B27  35B=$B35"

stage "5/8  27B deep (64k): MLX blocks 0,$B27 + GGUF"
$PY -u bench_mlx.py --model qwen3.6-27b --phase speed \
  --tiers deep --blocks 0,$B27 --kv-bits 0,8 >> "$L/mlx_27b.log" 2>&1
python3 -u bench.py --model qwen3.6-27b --quant q8 --phase speed \
  --tiers deep --kv-quant none,q8_0 >> "$L/gguf_27b.log" 2>&1
log "  exit=$?"

stage "6/8  35B deep (64k): MLX blocks 0,$B35 + GGUF"
$PY -u bench_mlx.py --model qwen3.6-35b-a3b --phase speed \
  --tiers deep --blocks 0,$B35 --kv-bits 0,8 >> "$L/mlx_35b.log" 2>&1
python3 -u bench.py --model qwen3.6-35b-a3b --quant q8 --phase speed \
  --tiers deep --kv-quant none,q8_0 >> "$L/gguf_35b.log" 2>&1
log "  exit=$?"

python3 bench.py --consolidate-only >> "$L/exhaustive.log" 2>&1

# --------------------------------------------------- 4. full phase for quality judging
stage "7/8  full-length phase (MTP off, fp16 KV, shallow) for judging"
$PY -u bench_mlx.py --model qwen3.6-27b     --phase full >> "$L/mlx_27b.log" 2>&1
$PY -u bench_mlx.py --model qwen3.6-35b-a3b --phase full >> "$L/mlx_35b.log" 2>&1
python3 -u bench.py --model qwen3.6-27b     --quant q8 --phase full >> "$L/gguf_27b.log" 2>&1
python3 -u bench.py --model qwen3.6-35b-a3b --quant q8 --phase full >> "$L/gguf_35b.log" 2>&1
log "  exit=$?"

stage "8/8  consolidate + report"
python3 bench.py --consolidate-only >> "$L/exhaustive.log" 2>&1
python3 judge_export.py >> "$L/exhaustive.log" 2>&1
python3 report.py >> "$L/exhaustive.log" 2>&1

log "=== exhaustive run complete ==="
