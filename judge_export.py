#!/usr/bin/env python3
"""Export the canonical answers that need grading into results/judging/to_grade.json.

Quality is a function of (model x quant x question), not of MTP draft depth (which only changes
speed). So we grade ONE canonical answer per model x quant x question — the pass-1, MTP-off
(draft_n=0) run.

A judge (Claude/Opus or any external grader) reads this file and writes results/judging/scores.json
following RUBRIC.md.
"""
import argparse
import json
import os

import config as C

THINK_PREVIEW_CHARS = 2000


def load_recs():
    with open(C.RESULTS_JSON) as f:
        return json.load(f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--all", action="store_true",
                    help="export every distinct output, not just the canonical ones")
    args = ap.parse_args()

    os.makedirs(C.JUDGING, exist_ok=True)
    recs = [r for r in load_recs() if r.get("error") is None]
    # Canonical answers come from the full-length phase (8192 cap, off config).
    recs = [r for r in recs if r.get("phase") == "full" and (r.get("answer_text") or r.get("thinking_text"))]
    # Exclude the legacy mlx_lm.server runs. They share quant="mlx8" with the current
    # mlx_vlm runs, so without this the grader would score stale outputs produced by a
    # harness with a corrupted thinking/answer split — and would silently win the dedupe
    # below by appearing earlier in the file.
    recs = [r for r in recs if r.get("runtime") != "mlx"]

    items, seen = [], set()
    for r in recs:
        model = r.get("model", "qwen3.6-27b")
        key = (model, r["quant"], r["question_id"])
        if key in seen:
            continue
        seen.add(key)
        from questions import QUESTION_BY_ID
        q = QUESTION_BY_ID[r["question_id"]]
        think = r.get("thinking_text", "") or ""
        items.append({
            "model": model,
            "quant": r["quant"],
            "question_id": r["question_id"],
            "category": r["category"],
            "question": q["user"],
            "thinking_tokens": r.get("thinking_tokens"),
            "answer_tokens": r.get("answer_tokens"),
            "thinking_preview": think[:THINK_PREVIEW_CHARS] + ("..." if len(think) > THINK_PREVIEW_CHARS else ""),
            "answer": r.get("answer_text", ""),
        })

    rubric_path = os.path.join(C.REPO, "RUBRIC.md")
    out = {
        "instructions": (
            "Score each item per RUBRIC.md on correctness, depth, instruction_following, "
            "and practicality (1-10 each), compute overall, add a one-line rationale, and "
            "write results/judging/scores.json with the same model+quant+question_id keys."
        ),
        "rubric_file": rubric_path,
        "schema_example": {
            "judge": "claude-opus-4-8", "rubric_version": 1,
            "scores": [{
                "model": "qwen3.6-27b", "quant": "q6", "question_id": "q2_coding",
                "correctness": 8, "depth": 7, "instruction_following": 8,
                "practicality": 8, "overall": 7.8, "rationale": "..."
            }],
        },
        "n_items": len(items),
        "items": items,
    }
    path = os.path.join(C.JUDGING, "to_grade.json")
    with open(path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"wrote {len(items)} items -> {path}")


if __name__ == "__main__":
    main()
