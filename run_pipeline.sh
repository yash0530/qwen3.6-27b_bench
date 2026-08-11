#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/Users/yash/Desktop/Programming/local_llm_bench/results/pipeline.log"
echo "=== Pipeline started at $(date) ===" > "$LOG_FILE"

# 1. Wait for GGUF benchmark process (PID 27430) to finish
echo "[$(date)] Waiting for GGUF benchmark process (PID 27430) to finish..." >> "$LOG_FILE"
while kill -0 27430 2>/dev/null; do
    sleep 10
done
echo "[$(date)] GGUF benchmark process finished." >> "$LOG_FILE"

# 2. Run MLX benchmarks
echo "[$(date)] Starting MLX benchmarks..." >> "$LOG_FILE"
.mlxenv/bin/python bench_mlx.py --model qwen3.6-35b-a3b >> "$LOG_FILE" 2>&1
echo "[$(date)] MLX benchmarks finished." >> "$LOG_FILE"

# 3. Generate final reports
echo "[$(date)] Generating final HTML and Markdown reports..." >> "$LOG_FILE"
python3 report.py >> "$LOG_FILE" 2>&1
echo "[$(date)] Reports generated successfully." >> "$LOG_FILE"

echo "=== Pipeline completed successfully at $(date) ===" >> "$LOG_FILE"
