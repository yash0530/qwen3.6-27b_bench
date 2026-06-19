#!/usr/bin/env bash
# Convenience runner: full benchmark then report. For the long run, prefer launching
# `python3 bench.py` in the background and calling report.py afterwards.
set -euo pipefail
cd "$(dirname "$0")"

MODE="${1:-full}"   # full | smoke
if [[ "$MODE" == "smoke" ]]; then
  python3 bench.py --smoke
else
  python3 bench.py
fi
python3 report.py
echo "Done. See REPORT.md and results/."
