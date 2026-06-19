# Judging rubric (v1)

The judge is **Claude/Opus** (this Claude Code session, a future session, or any pluggable
grader). Quality is scored per **(quant × question)** — MTP draft depth only affects speed,
so the canonical MTP-off pass-1 answer represents each quant×question.

## Criteria (score each 1–10)

| Criterion | Weight | What it measures |
|---|---|---|
| `correctness` | 0.35 | Technical accuracy; **no hallucinated facts/APIs/numbers**. Penalize confidently wrong claims hard. |
| `depth` | 0.25 | Insight and completeness — does it go beyond the obvious, cover edge cases / trade-offs? |
| `instruction_following` | 0.20 | Did it address **every** part of the multi-part prompt in the requested form? |
| `practicality` | 0.20 | Concrete, actionable, runnable — not hand-wavy. |

`overall = 0.35*correctness + 0.25*depth + 0.20*instruction_following + 0.20*practicality`
(If `overall` is omitted, `apply_scores.py` computes it from the weights.)

## How to grade (pluggable workflow)

1. `python3 judge_export.py` → `results/judging/to_grade.json` (15 canonical items; `--all` for every output).
2. Read each item's `question` + `answer` (+ `thinking_preview`), score against the table above.
3. Write `results/judging/scores.json`:

```json
{
  "judge": "claude-opus-4-8",
  "rubric_version": 1,
  "scores": [
    {"quant": "q6", "question_id": "q2_coding",
     "correctness": 8, "depth": 7, "instruction_following": 8, "practicality": 8,
     "overall": 7.75, "rationale": "Correct token-bucket; minor: no jitter cap."}
  ]
}
```

4. `python3 apply_scores.py` → merges `judge` onto every matching run in `results.json`.
5. `python3 report.py` → regenerates charts incl. quality + quality-vs-speed.

To re-grade later (different judge / criteria), bump `rubric_version`, drop in a new
`scores.json`, and re-run steps 4–5. An external/API judge just needs to emit the same
`scores.json` schema.
