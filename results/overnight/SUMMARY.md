# Overnight results — MLX + MTP + KV cache without crashes
_2026-08-22 02:20–07:00, M5 Pro / 64 GB. Full log: STAGES.md. Rails held all night: engine asserted before every run, per-turn RAM rail, ctx ≤65536, panic baseline 4 → 4._

## TL;DR

**Goal config achieved and survived the night: `mlx4` + checkpoint-only APC patch + `APC_EXACT_CACHE_ENTRIES=2` + MTP drafter (`LLM_SPEC=1`) + stripVolatile v2.**
14 consecutive real Claude Code turns with **zero panics** (the combo that panicked the machine twice on Aug 15), median warm turn **41 s**, fastest full agentic turn **9 s** (~3× faster than the same config without MTP). The binding constraint is no longer crashes — it's transient RAM headroom (~15–20 GB metal-heap growth during active turns, released at idle; my 25%/22% rails tripped three times and every trip was correct).

## 1. Run table

| # | Config | Turns | Panics | Median / best turn | Warm:cold requests | Verdict |
|---|---|---|---|---|---|---|
| A | llama.cpp :8091, GGUF UD-Q4_K_XL, `--cache-reuse`, MTP off | 2 (+29 req analyzed) | 0 | 550–900+ s | **0:29 — hybrid models get NO cross-request reuse** ("SWA or hybrid/recurrent memory") | **Rejected**: half of MLX decode speed (13.3 vs 27 t/s) AND no warmth |
| B | MLX4, APC=1, ENTRIES=2, MTP off, unpatched | 15/15 | 0 | 134 / 42 s | 43:12 (colds = harness volatility) | Passed |
| C | B + checkpoint-only patch | 15/15 | 0 | (see notes) | 43:12 | Passed |
| D | **C + MTP (GOAL)** | **14/14** (+1 rail-aborted pre-check) | **0** | **41 / 9 s** | warm throughout after fix below | **ADOPTED** |
| K | D + `--kv-bits 8` | 1 valid | 0 | — | — | **Rejected**: streaming tool-calls truncate — no `finish_reason` chunk ever sent; non-streaming fine. Upstream bug. |

## 2. Two root causes found & fixed tonight

1. **Periodic mid-prompt mutation killed warm cache every ~5th request.** Claude Code injects/moves a *"The task tools haven't been used recently…"* reminder inside the system prompt (and swaps its position with an adjacent `<system-reminder>` block between renders) — divergence at 46–79% depth ⇒ one 60–75 s full re-prefill each time. **Fixed** in `stripVolatile` (local-setup/scripts/llm-proxy.mjs): the nudge is now stripped like `<total_tokens>`. After the fix: **22 warm / 0 cold**, worst divergence 87.7% depth (safe tail extension).
2. **Dead double-store per cold prefill.** mlx_vlm exact mode stored the full prompt *and* a checkpoint at n−16; only the checkpoint can ever match a continuation. **Fixed** via `apply-latest-only-patch` (new, local-setup/scripts): gates both full-prompt stores behind `APC_SKIP_FULL_STORE` (default skip; `--revert` restores). Effect: one multi-GB GPU copy per cold prefill instead of two — the exact churn implicated in the Aug 15 IOGPU panics — and `ENTRIES=N` now holds N distinct states instead of N/2. (Not byte halving: checkpoint ≈ full size.)

## 3. Off-the-shelf research verdict (your "isn't this solved?" question)

| Piece | Verdict |
|---|---|
| llm-proxy.mjs | **Still needed**, but for a narrower reason: native `/v1/messages` exists everywhere now (llama.cpp PR #17570, LM Studio v0.4.1+), yet Claude Code's volatile prompt blocks need stripping, and this model's strict Jinja rejects Claude Code's second system-role message — the proxy's hoist fixes it. |
| llama.cpp for daily driving | No: hybrid-recurrent models get zero cross-request prefix reuse on build 9620, and decode is ~½ MLX speed (bench REPORT.md: best GGUF 13.5–14.8 t/s vs mlx4 27.2). |
| LM Studio | Untested (deprioritized — its mlx-engine has disk-backed KV checkpoints, the right design, but known Anthropic-endpoint cache-reset bug #327). Still worth a trial sometime. |

## 4. State left behind

- All engines stopped, 90% free, uptime continuous, panic files unchanged (4).
- **Changed files (uncommitted, review then commit):**
  - `local-setup/scripts/llm-proxy.mjs` — stripVolatile v2 (task-tools nudge).
  - `local-setup/scripts/apply-latest-only-patch` — NEW patch tool (validated apply→compile→revert).
  - Patch currently APPLIED to `.mlxenv/.../mlx_vlm/generate/{dispatch,ar}.py` (backups `.orig` alongside; `--revert` to undo).
  - `local_llm_bench/results/overnight/` — all logs, CSVs, dumps, harness scripts.
- GGUF downloaded: `~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf` (16 GB; keep or delete).

## 5. Morning quickstart

```bash
llm-serve start mlx4          # starts with LLM_APC=1 LLM_APC_ENTRIES=6 default — override:
LLM_APC=1 LLM_APC_ENTRIES=2 LLM_SPEC=1 LLAMA_PORT=8093 PROXY_PORT=8791 llm-serve start mlx4
# then claude local qwen38_27 --bits 4   (or point ANTHROPIC_BASE_URL at :8791)
```
For the standard ports make llm-serve's defaults match the adopted config (entries 6→2; consider exporting `APC_SKIP_FULL_STORE=1` and wiring `--kv-bits` off).

## 6. Open items

- RAM headroom: sustained sessions decay toward rail thresholds; recovers at idle. Next levers: `ENTRIES=1`, upstream fix for kv-bits streaming bug (halves snapshot bytes properly), or accept periodic idle gaps.
- Quality grading 4/6/8-bit still never done (only reason mlx8 was default).
- Report kv-bits streaming truncation upstream to mlx_vlm.
- Incidental: `~/.zshrc:116` hardcodes `OPENROUTER_API_KEY` (contradicts 6cad59b's env-only intent).
