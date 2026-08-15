#!/usr/bin/env python3
"""Cross-arm parity checks. Run before trusting any GGUF-vs-MLX comparison.

Motivation: the harness had a bug where the MLX arm generated with reasoning disabled
while the GGUF arm reasoned, because `enable_thinking` was passed to stream_generate but
not to the chat template (which is what actually opens the <think> block). Both arms ran
green, produced plausible tok/s, and were silently measuring different work for ~250 runs.

Every check below asks the same question in a different way: *are the two arms comparable*,
not *did each arm run*. A benchmark that only verifies the latter will happily compare
apples to oranges at full speed.

Exit code 1 if any check fails, so this can gate a report.

  python3 validate_parity.py
"""
import collections
import json
import os
import sys

import config as C

FAILURES = []
WARNINGS = []


def check(name, ok, detail=""):
    print(f"  [{'PASS' if ok else 'FAIL'}] {name}" + (f" — {detail}" if detail else ""))
    if not ok:
        FAILURES.append(name)


def warn(name, detail):
    print(f"  [WARN] {name} — {detail}")
    WARNINGS.append(name)


def load():
    recs = []
    with open(C.RESULTS_JSONL) as f:
        for ln in f:
            try:
                recs.append(json.loads(ln))
            except Exception:
                pass
    return [r for r in recs if r.get("error") is None]


def arm(r):
    """Which arm a record belongs to, or "legacy" if it cannot be compared.

    Records with no `runtime` predate the harness rewrite, and therefore predate the
    split_think fix that stopped truncated reasoning being filed as the *answer*. They
    report thinking_tokens == 0 for runs that did reason, so counting them as GGUF makes
    a correct arm look like it stopped reasoning — a failure in the data's history rather
    than in the comparison under test. They are excluded, not repaired: the raw text is
    still there for anyone who wants to re-derive them.
    """
    rt = r.get("runtime")
    if rt == "mlx_vlm":
        return "mlx"
    if rt == "gguf":
        return "gguf"
    return "legacy"


def reasoned(r):
    """Did this run actually reason?

    Prefer the recorded token split over sniffing the text. The heuristic below was
    calibrated on Qwen 3.6's reasoning openings and reports 0% on Qwen 3.8, whose traces
    start differently — which looks exactly like a broken arm when nothing is wrong.
    thinking_tokens is written by both harnesses from the same split_think(), so it is
    directly comparable across engines in a way that prose sniffing is not.
    """
    tt = r.get("thinking_tokens")
    if tt is not None:
        return tt > 0
    return looks_like_reasoning(r.get("raw_text"))


def looks_like_reasoning(raw):
    """Qwen emits reasoning without an opening tag (the template opens it in the prompt).

    So presence of reasoning is inferred from either a closing tag or the characteristic
    opening of a reasoning trace, not from a <think> tag that never appears in output.
    """
    if not raw:
        return False
    if "</think>" in raw:
        return True
    head = raw[:200].lower()
    return any(s in head for s in ("thinking process", "let me think", "first, i need",
                                   "let's break", "step 1", "**understand"))


def main():
    recs = [r for r in load() if arm(r) != "legacy"]
    if not recs:
        print("no records"); return 1
    print(f"records under test: {len(recs)}\n")

    # 1. Reasoning parity — the check that was missing.
    print("1. Reasoning engagement (both arms must reason, or neither)")
    models = sorted({r.get("model") or "?" for r in recs})
    for model in models:
        for phase in ("speed", "full"):
            rates = {}
            for a in ("gguf", "mlx"):
                rows = [r for r in recs if arm(r) == a and r.get("phase") == phase
                        and (r.get("model") or "?") == model and r.get("raw_text")]
                if not rows:
                    continue
                rates[a] = sum(1 for r in rows if reasoned(r)) / len(rows)
            if len(rates) == 2:
                gap = abs(rates["gguf"] - rates["mlx"])
                check(f"{model} {phase}: gguf={rates['gguf']:.0%} mlx={rates['mlx']:.0%}",
                      gap <= 0.15, f"gap {gap:.0%} (must be <=15%)")
            elif rates:
                warn(f"{model} {phase}: only one arm present", str(rates))

    # 2. Identical prompt depth per tier — both arms must see the same bytes.
    # Grouped per model, not just per tier: once more than one model family is in
    # results.jsonl, averaging a tier across models compares a Qwen 3.6 prompt to a
    # Qwen 3.8 one and the number stops meaning anything.
    print("\n2. Prompt length parity per model and tier")
    by = collections.defaultdict(dict)
    for r in recs:
        if r.get("prompt_n"):
            key = (r.get("model") or "?", r.get("prompt_tier") or "shallow")
            by[key].setdefault(arm(r), []).append(r["prompt_n"])
    for (model, tier), d in sorted(by.items()):
        if len(d) < 2:
            continue
        g = sum(d["gguf"]) / len(d["gguf"]); m = sum(d["mlx"]) / len(d["mlx"])
        rel = abs(g - m) / max(g, m)
        check(f"{model} {tier}: gguf={g:.0f} tok, mlx={m:.0f} tok", rel <= 0.02,
              f"differ by {rel:.1%} (must be <=2%)")

    # 3. Answers must exist where generation completed.
    print("\n3. Completed generations produce a non-empty answer")
    for a in ("gguf", "mlx"):
        rows = [r for r in recs if arm(r) == a and r.get("phase") == "full"]
        if not rows:
            continue
        empty = [r for r in rows if not (r.get("answer_text") or "").strip()]
        # Separate the two ways an answer can be missing. Hitting the token cap while
        # still reasoning is a budget finding about the model, not a broken harness;
        # an empty answer that finished normally is a real defect.
        capped = [r for r in empty
                  if (r.get("thinking_tokens") or 0) >= (r.get("n_predict_cap") or 0) - 2]
        broken = [r for r in empty if r not in capped]
        check(f"{a}: {len(rows) - len(empty)}/{len(rows)} full-phase runs have an answer",
              not broken, f"{len(broken)} empty for reasons other than the token cap")
        if capped:
            warn(f"{a}: {len(capped)} full-phase runs hit the cap mid-reasoning",
                 "raise FULL_N_PREDICT or lower reasoning_effort before grading these")

    # 4. Speculative rows must carry acceptance.
    print("\n4. Acceptance recorded for every speculative run")
    for a in ("gguf", "mlx"):
        rows = [r for r in recs if arm(r) == a and (r.get("draft_n") or 0) > 0
                and r.get("phase") == "speed"]
        if not rows:
            continue
        missing = [r for r in rows if r.get("acceptance_rate") is None]
        check(f"{a}: {len(rows) - len(missing)}/{len(rows)} MTP rows have acceptance",
              not missing, f"{len(missing)} missing")

    # 5. Cold prefill on both sides (no cross-question cache reuse).
    print("\n5. Prompt-cache isolation")
    leaks = [r for r in recs if r.get("cache_leak_tokens")]
    check(f"no cached-token leakage across questions", not leaks, f"{len(leaks)} leaked")

    # 6. Degenerate speculative configs (block_size 1 drafts nothing).
    print("\n6. No degenerate speculative configs")
    deg = [r for r in recs if arm(r) == "mlx" and r.get("draft_block_size") == 1]
    check("no mlx draft_block_size=1 rows", not deg, f"{len(deg)} present")

    # 7. Sampling parity — both arms must serve a model with the same knobs.
    # Qwen 3.8 wants temp 1.0 in thinking mode where 3.6 wanted 0.6, so this is now a
    # per-model value and therefore something that can drift between arms.
    print("\n7. Sampling parity per model")
    samp = collections.defaultdict(dict)
    for r in recs:
        if r.get("temp") is not None:
            samp[r.get("model") or "?"].setdefault(arm(r), set()).add(
                (r.get("temp"), r.get("top_p"), r.get("top_k")))
    for model, d in sorted(samp.items()):
        if len(d) < 2:
            continue
        check(f"{model}: gguf={sorted(d['gguf'])} mlx={sorted(d['mlx'])}",
              d["gguf"] == d["mlx"], "arms sampled differently")

    # 8. Rendered-prompt parity — the strongest available form of check 2.
    # llama-server renders the chat template itself via /apply-template; the MLX arm
    # renders locally. Comparing the resulting bytes catches template-kwarg drift
    # (Qwen 3.8's reasoning_effort, enable_thinking) that a token count can miss.
    # Warn rather than fail: the two renderers may legitimately differ on BOS handling,
    # and check 2 remains the hard gate on prompt depth.
    print("\n8. Rendered prompt parity (byte-level, advisory)")
    shas = collections.defaultdict(dict)
    for r in recs:
        if r.get("prompt_sha256"):
            key = (r.get("model") or "?", r.get("prompt_tier") or "shallow",
                   r.get("question_id"))
            shas[key].setdefault(arm(r), set()).add(r["prompt_sha256"])
    compared = matched = 0
    for key, d in shas.items():
        if len(d) < 2:
            continue
        compared += 1
        if d["gguf"] == d["mlx"]:
            matched += 1
    if compared:
        if matched == compared:
            check(f"{matched}/{compared} prompts byte-identical across arms", True)
        else:
            warn(f"{matched}/{compared} prompts byte-identical across arms",
                 "differing renders — confirm reasoning_effort/enable_thinking match")
    else:
        print("   (no overlapping arms with recorded prompt hashes yet)")

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print(f"all checks passed" + (f" ({len(WARNINGS)} warning(s))" if WARNINGS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
