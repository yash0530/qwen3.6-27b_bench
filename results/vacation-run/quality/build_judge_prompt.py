#!/usr/bin/env python3
"""Build one judging prompt file per question (all 3 quants, blind-labelled).

Usage: build_judge_prompt.py <question-number>
Writes judge.q<N>.txt and prints its path.
"""
import json, os, sys, re

QDIR = os.path.dirname(os.path.abspath(__file__))
RUBRIC = open(os.path.expanduser(
    "~/Desktop/Programming/local_llm_bench/RUBRIC.md")).read().split("## How to grade")[0]


def answer(quant, n):
    p = os.path.join(QDIR, f"{quant}.q{n}.md")
    if not os.path.exists(p):
        return None
    body = open(p).read()
    m = re.split(r"^## Answer\s*$", body, flags=re.M)
    return m[1].strip() if len(m) > 1 else body.strip()


def main():
    n = int(sys.argv[1])
    questions = json.load(open(os.path.join(QDIR, "questions.json")))
    q = questions[n - 1]

    parts = [
        "You are grading three answers produced by THE SAME 27B model served at three "
        "different quantization levels (8-bit, 6-bit, 4-bit). Your job is to detect "
        "quality loss from quantization.\n",
        "# Rubric\n", RUBRIC,
        "\n# Grading instructions\n",
        "- Score EACH of the three answers on all four criteria (1-10 integers) and compute "
        "`overall` with the weights above (one decimal).\n",
        "- ANSWER A is the 8-bit reference. Grade it on absolute merit. Then grade B and C on "
        "absolute merit too, but explicitly note any degradation relative to A: dropped "
        "requirements, weaker reasoning, hallucinated APIs/numbers, broken code, format "
        "violations, repetition, truncation.\n",
        "- Be strict and discriminating. Do NOT give all three the same score unless they are "
        "genuinely equivalent. Penalize confidently wrong claims hard.\n",
        "- IMPORTANT: all three answers were generated with the SAME 8192-token budget, and "
        "some were cut off mid-sentence when they hit that cap. Do NOT penalize the raw fact "
        "of a trailing cutoff. DO penalize an answer that squandered the shared budget on "
        "padding, repetition or restatement and therefore covered less of the question.\n",
        "- Output ONLY a JSON object, no markdown fence, no commentary, in this exact shape:\n",
        '{"A": {"correctness":N,"depth":N,"instruction_following":N,"practicality":N,'
        '"overall":N.N,"rationale":"one or two sentences"},'
        '"B": {...same keys...}, "C": {...same keys...}, '
        '"comparison":"2-3 sentences on how much quality B and C give up vs A"}\n',
        "\n# The question that was asked\n\n", q["user"], "\n",
    ]
    if n == 6:
        parts.append(
            "\nNOTE: this is an instruction-following test. Mechanically verify EVERY numbered "
            "constraint (exact headings, exact bullet/step counts, backticked artifacts, the "
            "`Verify` bullet, banned words, <300 words, the trailing CHECKLIST-END line) and "
            "score instruction_following on the count of violations.\n")

    for label, quant in (("A", "mlx8"), ("B", "mlx6"), ("C", "mlx4")):
        a = answer(quant, n)
        parts.append(f"\n\n=================== ANSWER {label} ===================\n\n")
        parts.append(a if a else "(MISSING — score all criteria 1 and say so.)")

    out = os.path.join(QDIR, f"judge.q{n}.txt")
    with open(out, "w") as f:
        f.write("".join(parts))
    print(out)


if __name__ == "__main__":
    main()
