#!/usr/bin/env bash
# Re-run the MLX arm with reasoning enabled, then the follow-on measurements.
# The GGUF data is unaffected and is not re-run.
set -uo pipefail
cd "$(dirname "$0")"
PY=.mlxenv/bin/python
L=results
log() { echo "[$(date +%T)] $*" | tee -a "$L/rerun.log"; }

log "=== MLX re-run (reasoning enabled) start ==="
log "STAGE 1/5  27B MLX shallow+agent"
$PY -u bench_mlx.py --model qwen3.6-27b --phase speed --tiers shallow,agent \
  --blocks 0,2,3,4,5 --kv-bits 0,8 >> "$L/mlx_27b_rerun.log" 2>&1; log "  exit=$?"
log "STAGE 2/5  35B MLX shallow+agent"
$PY -u bench_mlx.py --model qwen3.6-35b-a3b --phase speed --tiers shallow,agent \
  --blocks 0,2,3,4,5 --kv-bits 0,8 >> "$L/mlx_35b_rerun.log" 2>&1; log "  exit=$?"
log "STAGE 3/5  deep tier (both models, best block + off)"
$PY -u bench_mlx.py --model qwen3.6-27b --phase speed --tiers deep \
  --blocks 0,3 --kv-bits 0,8 >> "$L/mlx_27b_rerun.log" 2>&1; log "  exit=$?"
$PY -u bench_mlx.py --model qwen3.6-35b-a3b --phase speed --tiers deep \
  --blocks 0,2 --kv-bits 0,8 >> "$L/mlx_35b_rerun.log" 2>&1; log "  exit=$?"
log "STAGE 4/5  full phase for judging"
$PY -u bench_mlx.py --model qwen3.6-27b --phase full >> "$L/mlx_27b_rerun.log" 2>&1
$PY -u bench_mlx.py --model qwen3.6-35b-a3b --phase full >> "$L/mlx_35b_rerun.log" 2>&1
log "  exit=$?"
python3 bench.py --consolidate-only >> "$L/rerun.log" 2>&1

log "STAGE 5/5  warm-cache + concurrency"
for M in qwen3.6-27b qwen3.6-35b-a3b; do
  $PY -u bench_warmcache.py --arm mlx  --model "$M" --tier agent >> "$L/warm_$M.log" 2>&1
  python3 -u bench_warmcache.py --arm gguf --model "$M" --tier agent >> "$L/warm_$M.log" 2>&1
done
for M in qwen3.6-27b qwen3.6-35b-a3b; do
  for A in mlx gguf-mtp gguf-batch; do
    if [ "$A" = "mlx" ]; then R=$PY; else R=python3; fi
    $R -u bench_concurrency.py --arm "$A" --model "$M" --tier shallow \
      >> "$L/conc_${M}_${A}.log" 2>&1
    log "  conc $M $A exit=$?"
  done
done
python3 bench.py --consolidate-only >> "$L/rerun.log" 2>&1
python3 report.py >> "$L/rerun.log" 2>&1
log "=== rerun complete ==="
