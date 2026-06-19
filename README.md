# Qwen3.6-27B MTP benchmark (Q5 / Q6 / Q8) — Apple M5 Pro

Rigorous local benchmark of three quants of **Qwen3.6-27B-MTP** under llama.cpp, sweeping
the MTP draft depth (`--spec-draft-n-max` 1–8 plus an MTP-off baseline) across 5 realistic
prompts (model-selection/market-research, coding, architecture, financial reasoning,
debugging). Measures decode tok/s, prompt-processing tok/s, time-to-first-token, MTP draft
acceptance, and thinking vs answer token volume — then dumps JSON and renders charts.
Answer quality is graded by Claude/Opus as a pluggable judge.

## Design

- **Grid:** 3 quants × 9 MTP configs (`off`,1…8) × 5 questions × 2 passes = **270 runs**.
- Fixed seed + temp 0.6 → identical outputs across passes; the 2nd pass averages timing noise.
- One dedicated `llama-server` per (quant × draft-n); metrics pulled from the native
  `/completion` streaming `timings` (incl. `draft_n`/`draft_n_accepted`) + `/tokenize`.
- Crash-safe and resumable: every run is appended to `results/results.jsonl`; re-running skips
  completed `(pass, quant, draft_n, question)` cells.

## Run

```bash
python3 bench.py --smoke     # ~3–5 min end-to-end validation first
python3 bench.py             # full ~6–8 h run (run in background)
python3 report.py            # summary.json + charts/ + REPORT.md

# Judging (Claude/Opus):
python3 judge_export.py      # -> results/judging/to_grade.json
#   ...judge writes results/judging/scores.json per RUBRIC.md...
python3 apply_scores.py      # merge scores into results.json
python3 report.py            # regenerate with quality charts
```

`run.sh` chains bench + report. Requires `llama-server` (build ≥ 9620, MTP support) and the
three GGUFs under `~/Models/qwen3.6-27b-mtp-{q5,q6,q8}/`. Plotting uses matplotlib (already
present in system python3); orchestration is pure stdlib.

## Outputs (`results/`)

- `results.jsonl` / `results.json` — raw per-run records (incl. full answer + reasoning text, `judge`).
- `summary.json` — aggregates per quant and per quant×config + determinism stats.
- `charts/*.png`, `REPORT.md` — the readable report with the quant + draft-n + hosting recommendation.

See `RUBRIC.md` for the (pluggable) judging workflow and `config.py` for all knobs.
