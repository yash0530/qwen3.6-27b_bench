#!/usr/bin/env python3
"""How far does each quant drift from the reference quant's answers?

A rubric score says whether an answer is good. It does not say whether a quant is
answering the way the unquantized model would — a quant can lose real capability and
still score well on five questions, because the rubric has a ceiling and the questions
are few. This measures the other axis: agreement with a reference quant on the same
question, same prompt, same seed.

This is a *proxy*, and reported as one. The honest version of this measurement is KL
divergence against BF16 over hundreds of thousands of tokens, which needs logits the
serving APIs do not expose and hours of compute. What this does instead is compare the
generated text, which is downstream of the distribution and therefore strictly weaker
evidence — but it is evidence, it is cheap, and it is measured on exactly the workload
the model will actually serve.

Three numbers per (quant, question), all on the answer text with reasoning stripped:

  jaccard   — vocabulary overlap. Insensitive to ordering; catches gross divergence.
  prefix    — length of the common leading token run / reference length. At temperature
              near 0 an identical quant tracks the reference for a long prefix and then
              splits; this is where the split happens.
  len_ratio — generated length vs the reference. A quant that collapses into brevity or
              runs on is diverging even when its vocabulary overlaps.

Reported per quant as the mean across questions, never merged into the rubric score.

    python3 quant_agreement.py [--model qwen3.8-27b] [--reference q8]
"""
import argparse
import collections
import json
import os
import re
import statistics as st

import config as C

OUT = os.path.join(C.RESULTS, "quant_agreement.json")
WORD = re.compile(r"[a-z0-9_]+")


def tokens(text):
    return WORD.findall((text or "").lower())


def jaccard(a, b):
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 1.0
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 0.0


def common_prefix(a, b):
    n = 0
    for x, y in zip(a, b):
        if x != y:
            break
        n += 1
    return n


def load(model_id):
    recs = []
    with open(C.RESULTS_JSONL) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if (r.get("model") == model_id and not r.get("error") and not r.get("smoke")
                    and r.get("phase") == "full"):
                recs.append(r)
    return recs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-27b")
    ap.add_argument("--reference", default="q8",
                    help="quant every other quant is compared against")
    args = ap.parse_args()

    recs = load(args.model)
    if not recs:
        print(f"no full-phase records for {args.model}")
        return 1

    # (quant, question) -> answer text. Full phase is one pass per cell by construction.
    by = {}
    for r in recs:
        by[(r.get("quant"), r.get("question_id"))] = r.get("answer_text") or ""

    quants = sorted({q for q, _ in by})
    questions = sorted({q for _, q in by})
    if args.reference not in quants:
        print(f"reference quant {args.reference!r} not present; have {quants}")
        return 1

    rows = {}
    for quant in quants:
        j, p, lr = [], [], []
        for qid in questions:
            ref = by.get((args.reference, qid))
            cur = by.get((quant, qid))
            if not ref or not cur:
                continue
            rt, ct = tokens(ref), tokens(cur)
            if not rt:
                continue
            j.append(jaccard(rt, ct))
            p.append(common_prefix(rt, ct) / len(rt))
            lr.append(len(ct) / len(rt))
        if not j:
            continue
        rows[quant] = {
            "questions": len(j),
            "jaccard": round(st.mean(j), 4),
            "prefix": round(st.mean(p), 4),
            "len_ratio": round(st.mean(lr), 4),
        }

    print(f"agreement with {args.reference} ({args.model}, {len(questions)} questions)\n")
    print(f"{'quant':<10} {'jaccard':>8} {'prefix':>8} {'len_ratio':>10}")
    for quant, d in sorted(rows.items(), key=lambda kv: -kv[1]["jaccard"]):
        mark = "  <- reference" if quant == args.reference else ""
        print(f"{quant:<10} {d['jaccard']:>8.3f} {d['prefix']:>8.3f} "
              f"{d['len_ratio']:>10.3f}{mark}")

    payload = {"model": args.model, "reference": args.reference,
               "questions": questions, "agreement": rows}
    with open(OUT, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\nwrote -> {OUT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
