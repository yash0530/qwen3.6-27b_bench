#!/bin/bash
# mlx_stage.sh <start|stop> <entries> <spec 0|1> <patched 0|1>
# Runs the MLX stack on ALT ports (8093/8791) so the user's :8089/:8790 stack is untouched.
# Refuses to act while the 35b stack or client sessions are alive.
set -u
OV=/Users/yash/Desktop/Programming/local_llm_bench/results/overnight
LP=8093; PP=8791

guard() {
    if pgrep -f 'llama-server.*qwen3.6-35b' >/dev/null; then
        echo "REFUSE: 35b stack still running"; exit 1
    fi
    local n=$(netstat -an | grep '127.0.0.1.8790' | grep ESTABLISHED | wc -l | tr -d ' ')
    if (( n > 0 )); then
        echo "REFUSE: $n clients still connected to :8790"; exit 1
    fi
}

case ${1:?start|stop} in
start)
    guard
    ENT=${2:-2}; SPEC=${3:-0}; PATCHED=${4:-0}
    if (( PATCHED )); then
        /Users/yash/Desktop/Programming/local-setup/scripts/apply-latest-only-patch
    else
        /Users/yash/Desktop/Programming/local-setup/scripts/apply-latest-only-patch --revert 2>/dev/null | tail -2
    fi
    LLAMA_PORT=$LP PROXY_PORT=$PP LLM_APC=1 LLM_APC_ENTRIES=$ENT \
        LLM_APC_DISK= LLM_SPEC=$SPEC llm-serve start mlx4
    sleep 2
    SRV=$(cat ~/.local/state/local-llm/server.pid)
    echo "--- engine assertion ---"
    ps -o command= -p "$SRV" | grep -oE 'mlx_vlm.server|llama-server'
    ps -o command= -p "$SRV" | tr ' ' '\n' | grep -A1 '^--model$' | tail -1
    ps eww -p "$SRV" | tr ' ' '\n' | grep -E '^APC_ENABLED|^APC_EXACT_CACHE_ENTRIES|^APC_SKIP_FULL_STORE' || true
    ;;
stop)
    guard
    LLAMA_PORT=$LP PROXY_PORT=$PP llm-serve stop || true
    ;;
esac
