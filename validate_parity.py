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
    rt = r.get("runtime")
    if rt == "mlx_vlm":
        return "mlx"
    if rt in (None, "gguf"):
        return "gguf"
    return "legacy"


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
    for phase in ("speed", "full"):
        rates = {}
        for a in ("gguf", "mlx"):
            rows = [r for r in recs if arm(r) == a and r.get("phase") == phase
                    and r.get("raw_text")]
            if not rows:
                continue
            rates[a] = sum(1 for r in rows if looks_like_reasoning(r["raw_text"])) / len(rows)
        if len(rates) == 2:
            gap = abs(rates["gguf"] - rates["mlx"])
            check(f"{phase}: reasoning rate gguf={rates['gguf']:.0%} mlx={rates['mlx']:.0%}",
                  gap <= 0.15, f"gap {gap:.0%} (must be <=15%)")
        elif rates:
            warn(f"{phase}: only one arm present", str(rates))

    # 2. Identical prompt depth per tier — both arms must see the same bytes.
    print("\n2. Prompt length parity per tier")
    by = collections.defaultdict(dict)
    for r in recs:
        if r.get("prompt_n"):
            by[(r.get("prompt_tier") or "shallow")].setdefault(arm(r), []).append(r["prompt_n"])
    for tier, d in sorted(by.items()):
        if len(d) < 2:
            continue
        g = sum(d["gguf"]) / len(d["gguf"]); m = sum(d["mlx"]) / len(d["mlx"])
        rel = abs(g - m) / max(g, m)
        check(f"{tier}: gguf={g:.0f} tok, mlx={m:.0f} tok", rel <= 0.02,
              f"differ by {rel:.1%} (must be <=2%)")

    # 3. Answers must exist where generation completed.
    print("\n3. Completed generations produce a non-empty answer")
    for a in ("gguf", "mlx"):
        rows = [r for r in recs if arm(r) == a and r.get("phase") == "full"]
        if not rows:
            continue
        empty = [r for r in rows if not (r.get("answer_text") or "").strip()]
        check(f"{a}: {len(rows) - len(empty)}/{len(rows)} full-phase runs have an answer",
              not empty, f"{len(empty)} empty")

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

    print()
    if FAILURES:
        print(f"FAILED {len(FAILURES)} check(s): {', '.join(FAILURES)}")
        return 1
    print(f"all checks passed" + (f" ({len(WARNINGS)} warning(s))" if WARNINGS else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
