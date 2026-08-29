# mlx6 — q1 `q1_model_selection` (market_research)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=663.3s decode=12.4 tok/s prefill=347.2 tok/s gen_tokens=8192 prompt_tokens=260 finish=length
- thinking chars: 19788

## Prompt

I have a MacBook M5 Pro with 64 GB of unified RAM. I want to run the best open-source model that fits in about 45 GB of RAM, to act as a strong Sonnet-4.6 replacement AND as the main brain of a financial-analysis research engine (overnight ETL of price/news/earnings, tripwire risk signals, a deterministic morning digest that an LLM narrates). Do the market research: what are my realistic open-weight options in mid-2026, how do they compare on reasoning, coding, long-context and tool-use, and which one would you pick and why? Then explain how to host it on this machine to maximize accuracy and performance (quantization, context, KV cache, speculative/MTP decoding, serving stack), and how to wire it into the research engine. Be concrete and opinionated.

## Answer

# Open-Weight LLM on M5 Pro 64GB: Market Research & Architecture

## 1. The Memory Budget Reality Check

Before picking a model, let's be precise about what "45 GB" actually buys you, because the KV cache is the silent killer here.

| Component | Budget |
|-----------|--------|
| macOS + WindowServer + daemons | ~10 GB |
| Overnight ETL processes (Python, DuckDB, network buffers) | ~3–5 GB |
| **Available for LLM runtime** | **~40–45 GB** |
| → Weights (quantized) | variable |
| → KV cache (int8) | variable |
| → Runtime overhead (MLX/llama.cpp) | ~1–2 GB |

So in practice you have **~38–43 GB for weights + KV cache combined**. This is the number that constrains everything below.

### KV cache math (the part everyone underestimates)

For a 72B dense model (≈80 layers, GQA 8 KV heads, head_dim 128):

```
KV per token (int8) ≈ 2 × 80 × 8 × 128 × 1 byte ≈ 160 KB
32K context → ~5 GB
64K context → ~10 GB
128K context → ~20 GB  (impossible within budget)
```

For a 40B dense model (≈64 layers, GQA 8 KV heads):
```
KV per token (int8) ≈ 128 KB
32K → ~4 GB
64K → ~8 GB
128K → ~16 GB  (still tight)
```

**Key takeaway:** You cannot run 128K context at any meaningful model size within this budget. You need to either cap at 32K and use RAG for multi-document workloads, or use a smaller model with more KV headroom.

---

## 2. The Mid-2026 Open-Weight Landscape

I'll be transparent: the following blends confirmed 2025 releases with high-confidence projections based on announced roadmaps, team trajectories, and compute scaling trends. Confidence is noted.

### Dense models in the 30–80B sweet spot

| Model (projected) | Params | Context | Strengths | Confidence |
|---|---|---|---|---|
| **Qwen 3.5 / 4, 72B** | 72B dense | 128K native | Best open coding + reasoning; strong tool-use; excellent EN prose | High (Qwen 3 already strong) |
| **Llama 5, 70B** | 70B dense | 128K+ | Strong general reasoning; huge ecosystem; likely 1M ctx variant | Medium (Meta's cadence) |
| **Qwen 3.5 / 4, 40B** | 40B dense | 128K+ | Very strong per-param; great for local; good tool-use | High |
| **Mistral Large 3, 70B** | 70B dense | 128K | Strong EU-compliance angle; good reasoning; solid tool-use | Medium |
| **Gemma 4, 40B** | 40B dense | 64K+ | Very clean training; good reasoning for size; lighter ecosystem | Medium |
| **Phi-5, 32B** | 32B dense | 128K | Best-in-class efficiency; strong reasoning/size ratio; weaker long-horizon | Medium-High |

### MoE models (total params / active params)

| Model (projected) | Total / Active | Context | Notes | Confidence |
|---|---|---|---|---|
| **Llama 4 Scout (confirmed)** | 109B / 17B | 10M (claimed) | Already exists; at Q2_K fits ~35 GB; active params low | High |
| **DeepSeek V4-lite (projected)** | 150B / 40B | 64K+ | Would be exceptional if released; at Q3_K ~56 GB (too big); Q2_K ~38 GB | Low-Medium |
| **Mistral Mixtral 3, 8x28B** | 168B / 44B | 128K | At Q3_K ~63 GB (too big); Q2_K ~42 GB (borderline) | Medium |
| **Qwen MoE 100B / 30B (projected)** | 100B / 30B | 128K+ | At Q3_K ~42 GB — fits. Active 30B is solid. | Medium |
| **Olmoe 2 MoE 80B / 25B** | 80B / 25B | 32K+ | At Q4_K ~43 GB — borderline. Smaller ecosystem. | Low-Medium |

### Reasoning-specialist models

| Model | Notes |
|---|---|
| **DeepSeek R2 (projected)** | Full V3-scale; won't fit. A 70B "R2-lite" is plausible but speculative. |
| **Qwen 3.5 Thinking (projected)** | Qwen 3 already has a "thinking" mode; a dedicated reasoning 72B is likely. |
| **Llama 5 Reasoning (projected)** | Meta has signaled chain-of-thought models; 70B variant plausible. |

---

## 3. Head-to-Head on Your Dimensions

Scoring 1–5 for a 70–80B dense model in mid-2026 (projected, based on trajectory):

| Dimension | Qwen 3.5/4 72B | Llama 5 70B | Mistral 70B | Qwen 3.5/4 40B |
|---|---|---|---|---|
| **Multi-step reasoning** | 5 | 4.5 | 4 | 4 |
| **Coding (Python/SQL)** | 5 | 4.5 | 4 | 4.5 |
| **Tool-use / function-c
