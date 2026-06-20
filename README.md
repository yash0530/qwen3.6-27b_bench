# Qwen3.6-27B local benchmark — Q5 / Q6 / Q8 (GGUF+MTP) + MLX, on Apple M5 Pro (64 GB)

Rigorous local benchmark of **Qwen3.6-27B** as a Claude-Sonnet-4.6 replacement and the brain
of a financial-research engine. Measures decode tok/s, prompt-processing speed, time-to-first-
token, MTP draft acceptance, and thinking-vs-answer token volume across 5 realistic prompts
(model-selection / coding / architecture / financial-reasoning / debugging), with answer
quality graded 1–10 by **Claude/Opus** as a pluggable judge. Hardware: **M5 Pro, 64 GB, 20-core
GPU (Metal 4)**.

---

## TL;DR / Recommendation

**Run Qwen3.6-27B at Q8_0 (GGUF) with MTP `--spec-draft-n-max 2` (or 3) on llama.cpp.**
It is the best quant on every axis that matters and there's **no trade-off**: same peak speed
as the lighter quants, the highest answer quality, the best MTP acceptance, and the lowest
latency. It fits losslessly in ~33 GB (well inside a 45 GB budget). Deep research found nothing
in the 45 GB envelope that beats it (see `RESEARCH_models.md`).

| Quant | Quality (Opus 1–10) | Peak tok/s | Opt `n` | Speedup vs off | Accept | TTFT | tok/answer |
|---|---|---|---|---|---|---|---|
| Q5_K_XL (19 GB) | 8.30 | 17.7 | 3 | 1.37× | 57% | 864 ms | 6,795 |
| Q6_K_XL (24 GB) | 8.50 | 17.5 | 3 | 1.61× | 57% | 856 ms | 6,781 |
| **Q8_0 (27 GB)** | **8.67** | **17.7** | **2** | **1.80×** | **68%** | **761 ms** | 6,703 |
| MLX-8bit (29 GB)* | - | 8.0 | off | - | - | 1076 ms | 6,468 |

*\*MLX run was concluded early on the final debugging prompt; tok/answer based on the 4 completed prompts.*

---

## Results & analysis (GGUF & MLX)

**Decode tok/s by MTP draft-n / Runtime** (mean over questions; n capped at 4 — beyond that acceptance and throughput fall off):

| quant | off | n1 | n2 | n3 | n4 |
|---|---|---|---|---|---|
| Q5 | 12.9 | 14.9 | 13.4 | **17.7** | 17.2 |
| Q6 | 10.8 | 15.3 | 14.1 | **17.5** | 16.8 |
| Q8 | 9.9 | 15.4 | **17.7** | 17.7 | 16.9 |
| MLX-8bit | 8.0 | - | - | - | - |

**Draft acceptance vs draft-n** (near-identical across quants): n1≈80% · n2≈68% · n3≈57% ·
n4≈49% (decays to ~28% by n8, which is why n≥5 wastes compute).

**Key findings**
1. **Speed is not a differentiator** — all three quants converge to ~17.5–17.7 tok/s once MTP
   is tuned, so the choice is decided by quality + RAM, both of which favor Q8.
2. **Q8 is the surprise winner on MTP** — slowest without MTP (9.9 tok/s) yet the largest
   speedup (1.80×), highest acceptance (68%), fastest prompt processing, lowest TTFT. MTP fully
   erases its size penalty: Q8 fidelity at Q5 speed.
3. **n=2–4 is the sweet spot; n≥5 backfires** — acceptance decay means extra drafts get
   rejected. **n=3 is a safe universal setting** (within ~5% of every quant's peak).
4. **Quality is small-but-consistent: Q8 ≥ Q6 ≥ Q5** (8.67 / 8.50 / 8.30). All three produce
   excellent senior-level work; Q8's edge came from completeness (e.g. a `maxDelayMs` backoff
   cap and catching synchronous loader throws that the lighter quants missed). The architecture
   answers were strongest across the board and independently reinvented this engine's exact
   "deterministic digest, LLM-narrates-never-fabricates, full provenance" design.
5. **Determinism caveat (llama.cpp #23302):** 6/60 MTP runs produced a different output than
   their MTP-off baseline (~10%). Harmless for chat; for bit-reproducible output run with
   `--spec-type none`.
6. **q1 cutoff limit:** the local model's market-research answers are capped on correctness —
   its training predates the real mid-2026 models (it recommends a speculative "Qwen3-32B" and
   even says "MTP not supported, skip", unaware it *is* an MTP model). Don't use it for
   current-events / model-landscape questions.

Hosting line:
```bash
llama-server -m ~/Models/qwen3.6-27b-mtp-q8/Qwen3.6-27B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -c 16384 -ngl 99 -fa on -np 1 --jinja --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```
Wire into the engine via an `openai_compat` provider profile pointing at `:8080/v1`.

---

## Alternative models (research) — see `RESEARCH_models.md`

Deep web research (35 sources) on every open-weight model that fits ~45 GB on Apple Silicon vs
the frontier (Sonnet 4.6 / Opus 4.5 / GPT-5.5). Bottom line: **Qwen3.6-27B ties Sonnet 4.6 on
SWE-bench Verified and beats it on MMLU-Pro (+8), GPQA (+14), AIME, and Terminal-Bench.** The
only fit-window alternatives are Qwen3.6-35B-A3B (MoE, speed fallback) and Gemma 4 31B
(low-hallucination hedge, but no MTP). Every model that would clearly beat Qwen3.6-27B
(DeepSeek V4, GLM-5, Opus-class) is too large for 45 GB. Open risk: **tool-calling reliability**
has no published open number — validate locally.

---

## MLX comparison (concluded)

Tested **MLX** to compare its performance against `llama.cpp`. Benchmarked `unsloth/Qwen3.6-27B-MLX-8bit` (served via `mlx_lm`, no MTP support) across both phases. The run was concluded early on the final debugging prompt due to slow throughput.

**Key Findings:**
1. **Significantly Slower than GGUF+MTP**: MLX 8-bit achieved a mean decode speed of **8.0 tok/s** without speculative decoding. This is slower than GGUF Q8_0's raw baseline (9.9 tok/s) and less than half of GGUF Q8_0 + MTP's performance (**17.7 tok/s**).
2. **High Latency**: MLX's time-to-first-token (TTFT) averaged **1076 ms**, compared to **745–761 ms** for Q8_0 under llama.cpp.
3. **Prompt Processing Bottleneck**: MLX managed ~148 tok/s prefill speed, whereas llama.cpp Q8_0 achieved **282 tok/s** (nearly double the prefill throughput).
4. **Conclusion**: For the 27B model on Mac hardware, the general-purpose MLX runtime is outmatched by the specialized, hand-optimized C++/Metal execution model of `llama.cpp` combined with speculative decoding.

---

## Design

Two-phase, single-pass design (kept the run to ~5 h while preserving every metric):
- **Speed phase** — full grid (3 quants × `off`,1–4 × 5 questions) at a short 1024-token cap.
  tok/s is cap-independent, so this measures speed/acceptance/TTFT fairly and fast.
- **Full phase** — `off` config × quant × question at a 12,288-token cap (15 runs) for complete
  answers (judging) and true thinking/answer token totals. Output is deterministic across
  draft-n, so one config suffices.
- One dedicated server per config; GGUF metrics from llama.cpp `/completion` `timings`
  (incl. `draft_n`/`draft_n_accepted`); MLX metrics timed from the streamed SSE + tokenizer.
- Crash-safe + resumable (append to `results/results.jsonl`, keyed on
  `phase,pass,quant,draft_n,question`). GGUF (`q5/q6/q8`) and MLX (`mlx8`) coexist with distinct
  labels — rerunning either does not disturb the other.

## Run

```bash
# GGUF (llama.cpp):
python3 bench.py --smoke      # quick pipeline check
python3 bench.py              # full two-phase run
python3 report.py             # summary.json + charts/ + REPORT.md

# MLX (separate harness, venv):
.mlxenv/bin/python bench_mlx.py --smoke
.mlxenv/bin/python bench_mlx.py        # speed + full phases for mlx8

# Judging (Claude/Opus):
python3 judge_export.py       # -> results/judging/to_grade.json
#   ...judge writes results/judging/scores.json per RUBRIC.md...
python3 apply_scores.py       # merge scores into results.json
python3 report.py             # regenerate with quality charts
```

Requires `llama-server` (build ≥ 9620, MTP) + the GGUFs under `~/Models/qwen3.6-27b-mtp-{q5,q6,q8}/`;
MLX needs the venv (`.mlxenv`, `mlx-lm`/`mlx-vlm`) + `~/Models/qwen3.6-27b-mlx-8bit`.

## Outputs (`results/`)

- `results.jsonl` / `results.json` — raw per-run records (full answer + reasoning text, `judge`).
- `summary.json` — per-quant and per-quant×config aggregates + determinism stats.
- `charts/*.png`, `REPORT.md` — readable report with charts + the recommendation.

See `RUBRIC.md` for the pluggable judging workflow, `RESEARCH_models.md` for the model research,
and `config.py` for all knobs.
