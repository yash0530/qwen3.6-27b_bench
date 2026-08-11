#!/usr/bin/env bash
set -euo pipefail

LOG_FILE="/Users/yash/Models/.qwen36_35b_download.log"
mkdir -p "/Users/yash/Models"
echo "=== Download started at $(date) ===" > "$LOG_FILE"

# HF python environment path
HF_BIN="/Users/yash/Desktop/Programming/local_llm_bench/.mlxenv/bin/python /Users/yash/Desktop/Programming/local_llm_bench/.mlxenv/bin/hf"

# 1. Download Qwen 3.6 35B A3B MTP GGUF - Q5
echo "[$(date)] Starting Q5 GGUF..." >> "$LOG_FILE"
$HF_BIN download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q5_K_XL.gguf --local-dir /Users/yash/Models/qwen3.6-35b-a3b-mtp-q5 >> "$LOG_FILE" 2>&1

# 2. Download Qwen 3.6 35B A3B MTP GGUF - Q6
echo "[$(date)] Starting Q6 GGUF..." >> "$LOG_FILE"
$HF_BIN download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-UD-Q6_K_XL.gguf --local-dir /Users/yash/Models/qwen3.6-35b-a3b-mtp-q6 >> "$LOG_FILE" 2>&1

# 3. Download Qwen 3.6 35B A3B MTP GGUF - Q8
echo "[$(date)] Starting Q8 GGUF..." >> "$LOG_FILE"
$HF_BIN download unsloth/Qwen3.6-35B-A3B-MTP-GGUF Qwen3.6-35B-A3B-Q8_0.gguf --local-dir /Users/yash/Models/qwen3.6-35b-a3b-mtp-q8 >> "$LOG_FILE" 2>&1

# 4. Download MLX Qwen 3.6 35B A3B - 8bit
echo "[$(date)] Starting MLX 8-bit..." >> "$LOG_FILE"
$HF_BIN download mlx-community/Qwen3.6-35B-A3B-8bit --local-dir /Users/yash/Models/qwen3.6-35b-a3b-mlx-8bit >> "$LOG_FILE" 2>&1

# 5. Download MLX Qwen 3.6 35B A3B - 4bit
echo "[$(date)] Starting MLX 4-bit..." >> "$LOG_FILE"
$HF_BIN download mlx-community/Qwen3.6-35B-A3B-4bit --local-dir /Users/yash/Models/qwen3.6-35b-a3b-mlx-4bit >> "$LOG_FILE" 2>&1

echo "=== Download finished successfully at $(date) ===" >> "$LOG_FILE"
