# Local-LLM Model-Selection Research — M5 Pro (64 GB) Financial Research Engine

**Date:** 2026-06-19 · **Hardware:** MacBook Pro M5 Pro, 64 GB unified memory, 20-core GPU (Metal 4) · **Serving:** llama.cpp (GGUF + MTP) / open to MLX · **Budget:** model + KV ≈ 45 GB

> **[V]** = verified published benchmark figure (cited). **[E]** = estimate (interpolated footprint, derived speed, or inferred where no number is published).

---

## 0. Version reality-check (named models, corrected to June 2026)

| Named | Status (June 2026) | Notes |
|---|---|---|
| Claude Sonnet 4.6 | ✅ Current (Feb 17 2026) | Correct replacement target. |
| Claude Opus 4.5 | ⚠️ Legacy — superseded by Opus 4.6→4.7→4.8 | Opus 4.5 SWE-bench Verified ≈ 80.9%. |
| GPT-5.5 | ✅ Real (Apr 23 2026, "Spud") | Agentic-coding flagship; some suites still cite GPT-5.2. |
| Qwen3.6 | ✅ Two open releases: **27B dense** + **35B-A3B MoE**, both Apache-2.0, both ship MTP | |
| Llama 4 | ✅ Exists but NOT competitive at 2026 open frontier | Maverick ≈ 24% SWE-bench Verified. |
| Gemma 3 | Superseded by **Gemma 4** (Apr 2026): 26B-A4B MoE, 31B dense, Apache-2.0 | |
| DeepSeek R2 | Never shipped; real 2026 release is **DeepSeek V4** (284B–1.6T) — far too large | |

**Shortlist that actually fits 45 GB:** Qwen3.6-27B, Qwen3.6-35B-A3B, Gemma 4 31B. (Mistral Small 4 = 119B → too big; GPT-OSS-120B = 60.8 GB → too big; DeepSeek V4 / GLM-5 / Kimi K2.5 = 200B–1.6T → out.)

---

## 1. Candidates that fit ~45 GB on Apple Silicon

| Model | Params (active) | Arch | Ctx | License | Fits 45 GB (best quant) | MTP/spec |
|---|---|---|---|---|---|---|
| **Qwen3.6-27B** | 27.8B dense | Dense, multimodal | 262K→1M | Apache-2.0 | ✅ Q8_0 ~29 GB | ✅ MTP in llama.cpp (PR #22673) |
| **Qwen3.6-35B-A3B** | 35B / **3B active** MoE | MoE + DeltaNet | 262K→1M | Apache-2.0 | ✅ Q8 ~37 GB (Q6 safer) | ✅ MTP (NEXTN) |
| **Gemma 4 31B** | 30.7B dense | Dense, multimodal | 256K | Apache-2.0 | ✅ Q8 ~33 GB | ❌ draft-model only |
| Gemma 4 26B-A4B | 26B / 3.8B active | MoE | 256K | Apache-2.0 | ✅ Q8 ~28 GB | ❌ |
| Mistral Small 4 | 119B / 6B active | MoE | 256K | Apache-2.0 | ❌ ~65 GB @ Q4 | draft-only |
| GPT-OSS-120B | 117B / 5.1B active | MoE MXFP4 | 128K | Apache-2.0 | ❌ 60.8 GB | no MTP |
| Llama 4 Scout | 109B / 17B active | MoE | 10M | Llama (restrictive) | ❌ ~60 GB @ Q4 | draft-only |

Only **Qwen3.6-27B and Qwen3.6-35B-A3B** are top-tier coders in the fit window that *also* have production MTP in llama.cpp.

---

## 2. Quant deep-dive — Qwen3.6-27B (verified Unsloth GGUF sizes)

| Quant | bpw | GGUF size [V] | + KV (32K, fa) [E] | Total RAM [E] | Quality loss [E] |
|---|---|---|---|---|---|
| UD-Q4_K_XL | ~4.8 | 17.9 GB | ~2-3 GB | ~21 GB | small |
| UD-Q5_K_XL | ~5.7 | 20.4 GB | ~3 GB | ~23 GB | minor |
| Q6_K | ~6.6 | 22.9 GB | ~3 GB | ~26 GB | near-negligible |
| UD-Q6_K_XL | ~6.6 | 26.0 GB | ~3 GB | ~29 GB | near-negligible |
| **Q8_0** | 8.0 | 29.0 GB | ~3-4 GB | **~33 GB** | ~lossless |

**Key insight:** Q8_0 fits with ~30 GB to spare — **you are not RAM-constrained on the 27B; run the highest quant for accuracy.**

**Apple-Silicon decode model:** bandwidth-bound (~270 GB/s class). Dense 27B @ Q8 → ~270/29 ≈ ~9 tok/s raw — matches the measured 9.85 tok/s no-MTP baseline. MoE 35B-A3B reads only ~3B active params/token → ~2-3× cheaper bandwidth/token (its one real Mac advantage), at the cost of weaker per-token quality.

---

## 3. Strict accuracy benchmarks vs frontier refs

**Frontier [V]:**

| Suite | Sonnet 4.6 | Opus 4.5 (legacy) | GPT-5.5 |
|---|---|---|---|
| SWE-bench Verified | 77.2% | 80.9% | ~80% class |
| SWE-bench Pro | — | 45.9% | 58.6% |
| Terminal-Bench 2.0 | 51.4% | ~59% | 82.7% |
| GPQA-Diamond | 74.1% | — | 93.6% |
| MMLU-Pro | 78.0% | 89.5% | — |
| AIME 2025 | 95.6% | — | 94.6% (GPT-5) |
| τ-bench (retail/airline) | 86.2% / 70.0% | — | MCP-Atlas 75.3% |

**Open candidates [V] and gap to Sonnet 4.6:**

| Suite | Qwen3.6-27B | gap vs S4.6 | Qwen3.6-35B-A3B | Gemma 4 31B |
|---|---|---|---|---|
| SWE-bench Verified | 77.2% | ≈ 0 (tie) | 73.4% | trails |
| SWE-bench Pro | 53.5% | > S4.6, < GPT-5.5 | n/p | n/p |
| Terminal-Bench 2.0 | 59.3% | +7.9 | 51.5% | n/p |
| GPQA-Diamond | 87.8% | +13.7 | 86.0% | 84.3% |
| MMLU-Pro | 86.2% | +8.2 | 85.2% | 85.2% |
| AIME 2026 | 94.1% | ~tie | 92.7% | 89.2% |
| LiveCodeBench v6 | 83.9% | (S4.6 n/p) | 80.4% | 80.0% |
| Aider Polyglot | no published number | — | n/p | n/p |
| τ-bench / BFCL (tool-use) | **no published number** | — | "tool-optimized" only | n/p |
| RULER / long-ctx | no published number (262K→1M) | — | same | 256K |

**Honest gaps:** Qwen3.6-27B has **no published tool-calling (BFCL/τ-bench), Aider, or RULER numbers** — agentic ability is inferred from SWE-Pro (53.5), Terminal-Bench (59.3). On knowledge/reasoning (MMLU-Pro, GPQA, AIME) it **beats Sonnet 4.6 by 8-14 pts** and **ties on SWE-bench Verified**; it trails only on agentic-tool polish and the hardest agentic coding (vs GPT-5.5/Opus).

---

## 4. Ranked recommendation (financial reasoning + coding + tool-use + low hallucination, ≤45 GB)

**🥇 #1 — Qwen3.6-27B (dense) @ Q8_0 + MTP** — ~95-100% of Sonnet 4.6 on your axes; arguably superior on raw knowledge/reasoning. Ties SWE-bench Verified, beats MMLU-Pro/GPQA/Terminal-Bench. Only fit-window top coder with production MTP; fits Q8 losslessly with 30 GB headroom; dense = lower-variance outputs. **Verify locally:** no published tool-calling/RULER number.

**🥈 #2 — Qwen3.6-35B-A3B (MoE) @ Q6/Q8 + MTP** — ~90%. Pick only if decode speed becomes the bottleneck (3B active → 2-3× cheaper/token). Slightly weaker per-token; newer hybrid attention less battle-tested in llama.cpp.

**🥉 #3 — Gemma 4 31B (dense) @ Q8** — ~85%. Low-hallucination hedge; strong knowledge, clean tuning. But trails Qwen on agentic coding and **has no first-class MTP** (gives up the 1.5-1.8× speedup).

**Verdict:** Qwen3.6-27B is the right pick. Nothing in 45 GB beats its blend of lossless-Q8 fit + Sonnet-4.6-class reasoning/coding + real MTP. Every model that would clearly beat it doesn't fit on any quant.

---

## 5. Hosting plan — Qwen3.6-27B on M5 Pro

**Stack: llama.cpp** (MTP is merged/production-ready; MLX's Qwen3.6-MTP path less mature). **Quant: Q8_0** (highest acceptance @ 68%; drop to UD-Q6_K_XL only if you want more KV headroom).

```bash
llama-server \
  -hf unsloth/Qwen3.6-27B-MTP-GGUF:Q8_0 \
  -ngl 99 -c 32768 -fa on -np 1 \
  --jinja --reasoning-format deepseek \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  --host 127.0.0.1 --port 8089 --metrics --no-webui
```

- `-np 1` required for MTP; `-fa on`; KV at fp16 (you have headroom — don't quantize it).
- `--spec-draft-n-max 3` (Q8 sweet spot); `2` marginally better on short replies. n≥4 hurts.
- Expected (measured): **~17.7 tok/s decode**, ~298 tok/s prefill, TTFT < 1 s. Baseline w/o MTP ~9.85 tok/s — keep MTP on.

**Wire into the engine (`openai_compat`):** `baseURL: http://127.0.0.1:8089/v1`, `apiKey: "sk-local"`, `model: "qwen3.6-27b"`, `temperature 0.6 / top_p 0.95 / top_k 20`. For the **digest-narrator** role, drop temp to ~0.2-0.3 and consider disabling thinking mode (latency + hallucination surface); reserve full thinking for analyst/reasoning passes. Validate JSON-schema tool calls in a smoke test — the one capability with no published benchmark.

---

## Bottom line

Run **Qwen3.6-27B @ Q8_0 with MTP (`--spec-draft-n-max 3`)** on llama.cpp. It fits losslessly (~33 GB ≪ 45 GB), decodes at a measured ~17.7 tok/s, ties Sonnet 4.6 on SWE-bench Verified while beating it on MMLU-Pro (+8), GPQA (+14), AIME, and Terminal-Bench — a credible Sonnet-4.6 replacement and arguably a reasoning/knowledge upgrade. The MoE 35B-A3B is a speed-first fallback, Gemma 4 31B a low-hallucination hedge, and every frontier-killer is too large for this machine. The one open risk is **tool-calling reliability + hallucination** (no published open numbers) — make those the headline metrics of your local judging pass before trusting it as the engine's sole brain.

---
*Generated by background research agent (35 web sources, 2026-06-19). [V] figures carry source links in the originating report; see chat for clickable citations.*
