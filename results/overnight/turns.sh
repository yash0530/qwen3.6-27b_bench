#!/bin/bash
# turns.sh <label> <n_turns> <direct|proxy> [port]
# Drives N real multi-turn Claude Code turns in a sandbox, with RAM rail per turn.
set -u
OV=/Users/yash/Desktop/Programming/local_llm_bench/results/overnight
source "$OV/lib.sh"

LABEL=${1:?label}; N=${2:?n}; MODE=${3:?direct|proxy}; PORT=${4:-8089}
SANDBOX="$OV/sandbox-$LABEL"
CSV="$OV/run-$LABEL.csv"
TURNLOG="$OV/turns-$LABEL.log"
CLAUDE=/Users/yash/.local/bin/claude

mkdir -p "$SANDBOX"
[[ -f "$CSV" ]] || echo "turn,wall_s,free_pct,rc" >> "$CSV"

if [[ "$MODE" == "proxy" ]]; then
    BASE="http://127.0.0.1:$PORT"; MODEL=qwen-local
else
    BASE="http://127.0.0.1:$PORT"; MODEL=qwen38-local
fi

run_claude() { # $1=prompt $2=continue?
    local cont=""; [[ "${2:-}" == "cont" ]] && cont="--continue"
    (cd "$SANDBOX" && ANTHROPIC_BASE_URL="$BASE" ANTHROPIC_AUTH_TOKEN=local \
        ANTHROPIC_MODEL="$MODEL" ANTHROPIC_SMALL_FAST_MODEL="$MODEL" \
        ANTHROPIC_DEFAULT_HAIKU_MODEL="$MODEL" \
        CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536 MAX_THINKING_TOKENS=0 \
        API_TIMEOUT_MS=1800000 CLAUDE_STREAM_IDLE_TIMEOUT_MS=1800000 \
        CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC=1 \
        perl -e 'alarm shift; exec @ARGV' 900 \
        "$CLAUDE" --dangerously-skip-permissions $cont -p "$1") \
        >> "$TURNLOG" 2>&1
}

i=0
while read -r prompt; do
    i=$((i+1)); (( i > N )) && break
    free=$(mem_rail) || { echo "[$(now)] ABORT: RAM rail tripped at turn $i" | tee -a "$TURNLOG"; break; }
    cont=""; (( i > 1 )) && cont=cont
    t0=$(date +%s)
    run_claude "$prompt" $cont
    rc=$?
    t1=$(date +%s)
    echo "$i,$((t1-t0)),$free,$rc" >> "$CSV"
    echo "[$(now)] turn $i wall=$((t1-t0))s rc=$rc" >> "$TURNLOG"
done < "$OV/prompts.txt"

echo "[$(now)] turns done: $LABEL" >> "$TURNLOG"
