#!/bin/bash
# Shared rails for overnight runs. Source this, don't execute it.
OV=/Users/yash/Desktop/Programming/local_llm_bench/results/overnight
PANIC_DIR=/Library/Logs/DiagnosticReports

now() { date "+%H:%M:%S"; }

mem_free_pct() {
    memory_pressure 2>/dev/null | sed -n 's/.*free percentage: \([0-9]*\)%.*/\1/p'
}

# Abort everything if RAM headroom is gone.
mem_rail() {
    local pct; pct=$(mem_free_pct)
    if [[ -z "$pct" ]]; then echo "[$(now)] MEM RAIL UNREADABLE" >> "$OV/rails.log"; return 1; fi
    if (( pct < 25 )); then
        echo "[$(now)] MEM RAIL TRIPPED: ${pct}% free" >> "$OV/rails.log"
        return 1
    fi
    echo "$pct"
}

panic_count() { ls "$PANIC_DIR"/panic-full-*.panic 2>/dev/null | wc -l | tr -d ' '; }

# Baseline is 4 panics (Aug 15 x2, Aug 16 x2). Any increase = machine rebooted since.
panic_check() {
    local c; c=$(panic_count)
    if (( c > 4 )); then
        echo "PANICS NOW $c (baseline 4) — machine crashed during run"
        ls -lt "$PANIC_DIR"/panic-full-*.panic | head -3
        return 1
    fi
    echo "no new panics ($c total)"
}

stage() {  # stage "<name>" "<config description>"
    echo "" >> "$OV/STAGES.md"
    echo "## $(date '+%Y-%m-%d %H:%M') — $1" >> "$OV/STAGES.md"
    echo "- config: $2" >> "$OV/STAGES.md"
    echo "- uptime: $(uptime)" >> "$OV/STAGES.md"
}
