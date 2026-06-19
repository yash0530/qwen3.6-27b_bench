#!/usr/bin/env python3
"""Merge results/judging/scores.json into results/results.json.

Attaches a `judge` block to every run sharing the scored (quant, question_id), so the
JSON dump carries quality alongside the speed metrics. Idempotent. After running this,
re-run report.py to regenerate the quality charts.
"""
import json
import os

import config as C

WEIGHTS = {"correctness": 0.35, "depth": 0.25, "instruction_following": 0.2, "practicality": 0.2}


def compute_overall(item):
    if "overall" in item and isinstance(item["overall"], (int, float)):
        return float(item["overall"])
    s = sum(item.get(k, 0) * w for k, w in WEIGHTS.items())
    return round(s, 2)


def main():
    sp = os.path.join(C.JUDGING, "scores.json")
    if not os.path.exists(sp):
        raise SystemExit(f"missing {sp} — produce it from to_grade.json first (see RUBRIC.md)")
    with open(sp) as f:
        scores = json.load(f)

    index = {}
    for item in scores.get("scores", []):
        item["overall"] = compute_overall(item)
        index[(item["quant"], item["question_id"])] = item

    with open(C.RESULTS_JSON) as f:
        recs = json.load(f)

    merged = 0
    for r in recs:
        key = (r.get("quant"), r.get("question_id"))
        if key in index:
            it = index[key]
            r["judge"] = {
                "judge": scores.get("judge", "claude-opus"),
                "rubric_version": scores.get("rubric_version", 1),
                "correctness": it.get("correctness"),
                "depth": it.get("depth"),
                "instruction_following": it.get("instruction_following"),
                "practicality": it.get("practicality"),
                "overall": it.get("overall"),
                "rationale": it.get("rationale", ""),
            }
            merged += 1

    with open(C.RESULTS_JSON, "w") as f:
        json.dump(recs, f, indent=2)
    # also persist normalized overalls back to scores.json
    with open(sp, "w") as f:
        json.dump(scores, f, indent=2)
    print(f"merged judge scores onto {merged} records in {C.RESULTS_JSON}")
    print("now re-run: python3 report.py")


if __name__ == "__main__":
    main()
