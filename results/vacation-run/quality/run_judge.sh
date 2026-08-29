#!/bin/bash
D=/Users/yash/Desktop/Programming/local_llm_bench/results/vacation-run/quality
A=~/.claude/plugins/cache/antigravity-cc/agy/0.4.1/scripts/agy-run.sh
for n in 1 2 3 4 5; do
  echo "=== judging q$n $(date +%H:%M:%S) ==="
  bash $A ask --model flash --dangerously-skip-permissions \
    -p="Read the file $D/judge.q$n.txt and follow the grading instructions inside it exactly. Be strict and discriminating: these are three quantizations of one model and you must surface real differences. Output ONLY the JSON object it asks for, no markdown fence, no commentary." \
    > $D/judge.q$n.out 2> $D/judge.q$n.err
  echo "exit=$? bytes=$(wc -c < $D/judge.q$n.out)"
done
echo "JUDGING DONE $(date +%H:%M:%S)"
