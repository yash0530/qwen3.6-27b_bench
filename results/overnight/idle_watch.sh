#!/bin/bash
# Exits 0 when the 35b stack has been idle >=20 min (or died); hard cap 5 h.
OV=/Users/yash/Desktop/Programming/local_llm_bench/results/overnight
LOG=~/.local/state/local-llm/logs/server.log
SRV_PID=$(pgrep -f 'llama-server.*qwen3.6-35b' | head -1)
idle=0; cap=0
while (( cap < 300 )); do
    sleep 60; cap=$((cap+1))
    if [[ -n "$SRV_PID" ]] && ! kill -0 "$SRV_PID" 2>/dev/null; then
        echo "$(date '+%H:%M:%S') server exited" >> "$OV/idle-watch.log"; exit 0
    fi
    cur=$(grep -oE 'task [0-9]+' "$LOG" 2>/dev/null | tail -1)
    if [[ "$cur" == "$last" ]]; then
        idle=$((idle+1))
    else
        idle=0; last="$cur"
    fi
    echo "$(date '+%H:%M:%S') last=$cur idle_min=$idle free=$(memory_pressure | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p')%" >> "$OV/idle-watch.log"
    if (( idle >= 20 )); then echo "$(date '+%H:%M:%S') IDLE 20min — proceeding" >> "$OV/idle-watch.log"; exit 0; fi
done
echo "$(date '+%H:%M:%S') cap reached while still busy" >> "$OV/idle-watch.log"
exit 2
