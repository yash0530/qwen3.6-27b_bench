# Quality grading — Qwen 3.8 27B MLX 8/6/4-bit

Run 2026-08-28 23:39 → 2026-08-29 02:11. Safe path throughout: `LLM_APC=0 LLM_SPEC=0`
(prompt cache off, drafter off), one server at a time, prompts 185–334 tokens.

**Rails: clean.** Panic-file count stayed at 1 (`panic-full-2026-08-28-224750`, the
baseline) for the whole run; uptime continuous from the 22:47 boot (3 h 24 m at the end);
zero `kIOGPUCommandBufferCallbackErrorOutOfMemory` in any server log. All servers stopped.

## Configuration

Identical across all three quants:

| | |
|---|---|
| temperature / top_p | 1.0 / 0.95 (`llm-serve model_temp()` for mlx*) |
| seed | 1234 (accepted by the server; see caveat below) |
| max_tokens | 8192 |
| thinking | on, `reasoning_effort=low` |
| serving | `mlx_vlm.server`, mlx-vlm 0.6.17, APC off, MTP off |

Two settings had to be changed from the brief, both forced by measurement:

- **max_tokens 4096 → 8192.** At 4096 with default (xhigh) effort, mlx8 q1 spent the
  *entire* budget on reasoning and returned **zero answer characters**
  (`finish_reason=length`, 12,535 chars of thinking, 0 of answer). Ungradable.
- **`reasoning_effort=low`.** Even at 8192 the model needs the reasoning trimmed to leave
  room for an answer. `low` is also what the agent path approximates (`llm-serve` defaults
  `LLM_EFFORT=medium`).

Even after both changes, q1 and q3 still hit the cap for all three quants (2/6 truncated
each). Because the budget is identical, this is fair across quants, and the judge was
explicitly told not to penalize a trailing cutoff — only budget *squandered* on padding.

## Scores (1–10, rubric-weighted overall)

Judged by Antigravity CLI (`agy ask --model flash`) against `RUBRIC.md`, one call per
question with all three answers labelled A/B/C (A = 8-bit reference).

| Question | Category | mlx8 | mlx6 | mlx4 |
|---|---|---|---|---|
| q1 model selection | market research | **8.8** | 4.2 | 4.8 |
| q2 coding | coding | 4.0 | 4.7 | **7.2** |
| q3 architecture | architecture | **9.2** | 6.7 | 6.6 |
| q4 finance reasoning | reasoning | 9.2 | **9.7** | 8.9 |
| q5 debugging | debugging | 9.2 | 7.1 | **10.0** |
| q6 instruction following | instructions | **9.2** | 8.4 | **9.2** |
| **mean** | | **8.27** | **6.80** | **7.78** |
| **median** | | **9.20** | **6.90** | **8.05** |

Per-criterion means:

| Criterion | mlx8 | mlx6 | mlx4 |
|---|---|---|---|
| correctness (0.35) | 7.83 | 7.00 | **8.00** |
| depth (0.25) | **8.50** | 6.50 | 7.67 |
| instruction following (0.20) | **9.00** | 7.50 | 7.83 |
| practicality (0.20) | **8.00** | 6.00 | 7.50 |

## Speed and token economy

| | mlx8 | mlx6 | mlx4 |
|---|---|---|---|
| decode tok/s (mean of 6) | 9.7 | 12.4 | **17.6** |
| decode spread | 9.7–9.7 | 12.3–12.6 | 17.5–17.8 |
| server RSS | 25.8 GB | 21.7 GB | **15.5 GB** |
| wall for all 6 questions | 62 min | 54 min | **34 min** |
| mean thinking chars | 7,884 | 12,053 | 9,042 |
| mean answer chars | 14,358 | 12,959 | 13,158 |
| fraction of output spent thinking | 0.43 | **0.54** | 0.47 |
| truncated (hit 8192) | 2/6 | 2/6 | 2/6 |

Decode rate was extremely stable within each quant — the ±0.1–0.3 tok/s spread means the
speed numbers are solid even though the quality numbers are not. **mlx4 is 1.81× mlx8 and
1.42× mlx6.**

## Verdict

On this evidence **mlx4 does not give up a quality margin large enough to justify its
1.81× speed and 10 GB memory penalty over mlx8** — it scored 7.78 mean against mlx8's
8.27, and it actually *beat* the 8-bit reference on three of six questions (coding,
debugging, instruction-following). The result ordering is non-monotonic — mlx6 came last
at 6.80, below the 4-bit — which is not a physically sensible quantization curve and is
the clearest signal in the table that with one sample per cell at temperature 1.0,
run-to-run sampling variance is larger than the quantization effect being measured. For
daily agent use the honest reading is that no gap here is large enough to care about, and
mlx4's speed advantage is real, measured and repeatable in a way the quality gap is not.

### Why these scores should not be over-read

- **n=1 per cell at temp 1.0.** The spread between best and worst quant on a single
  question reaches 5.2 points (q1), far larger than the 1.5-point spread between quant
  means. Differences of this size are dominated by which sample the model happened to draw.
- **The single biggest score swing is a sampling accident, not quantization.** mlx8 scored
  4.0 on q2 — its worst cell, and the reason its mean isn't higher — because it emitted a
  genuinely fatal bug. In its worker pool the success path does
  `successes.push(...); return;` inside the `while(true)` worker IIFE, so **every worker
  permanently exits after its first successful item**: with concurrency 4 and 100 items,
  only 4 ever run. I verified this by hand in `mlx8.q2.md`; the judge's call was correct.
  The 4-bit build produced a working pool for the same prompt. Nothing about 8-bit
  weights causes that — it is one bad draw.
- **The judge is lenient and not fully discriminating.** On q6 it gave all three 10/10 on
  instruction-following. I mechanically re-checked every constraint (exact headings, 4/7/3
  counts, backticked artifacts, the `Verify` bullet, banned words, <300 words, trailing
  `CHECKLIST-END`) and all three genuinely passed — so that call was right, but a grader
  that returns straight 10s is weak evidence either way.
- **`seed` is accepted but not verified to be honoured.** The server did not reject the
  field; nothing confirms `mlx_vlm.server` actually uses it for sampling. Do not assume
  these runs are reproducible.

### The one clean quality-adjacent signal

The most trustworthy quality difference in this run is not a rubric score but a token
statistic: **mlx6 spends materially more of its budget on reasoning** (54% of output
characters, vs 43% for mlx8 and 47% for mlx4; 12.1k mean thinking chars vs mlx8's 7.9k).
On q1 that cost it the answer — 19,788 chars of thinking left only 3,945 chars of answer
before the cap, against mlx8's 18,697 chars of answer, and the judge penalized the
resulting thin coverage hard (4.2). If that verbosity is a real property of the 6-bit
build rather than another sampling artifact, it is a genuine argument against mlx6 for
agent use, where budget spent thinking is latency the user waits through. That hypothesis
is worth one targeted re-run; it is not established by n=1.

### What would settle it

Three samples per cell at temp 1.0 (or one greedy sample per cell if the server honours
`temperature=0` without looping), scored blind. That is ~9 h of generation on this
machine at these rates — an overnight job, not an interactive one. Until then, treat the
speed table as measured fact and the quality table as a weak prior.

## Files

- `<quant>.q<N>.md` — 18 answers with per-request decode/prefill/token stats
- `<quant>.stats.json` — machine-readable per-question stats
- `questions.json` — the 6 prompts (repo's 5 from `questions.py` + one instruction test)
- `judge.q<N>.txt` / `.out` — judging prompts and raw verdicts
- `scores.json` — parsed score matrix with per-criterion detail
- `run_quant.py`, `build_judge_prompt.py`, `run_judge.sh` — the harness
