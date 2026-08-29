# mlx4 — q1 `q1_model_selection` (market_research)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=469.7s decode=17.5 tok/s prefill=370.9 tok/s gen_tokens=8192 prompt_tokens=260 finish=length
- thinking chars: 12216

## Prompt

I have a MacBook M5 Pro with 64 GB of unified RAM. I want to run the best open-source model that fits in about 45 GB of RAM, to act as a strong Sonnet-4.6 replacement AND as the main brain of a financial-analysis research engine (overnight ETL of price/news/earnings, tripwire risk signals, a deterministic morning digest that an LLM narrates). Do the market research: what are my realistic open-weight options in mid-2026, how do they compare on reasoning, coding, long-context and tool-use, and which one would you pick and why? Then explain how to host it on this machine to maximize accuracy and performance (quantization, context, KV cache, speculative/MTP decoding, serving stack), and how to wire it into the research engine. Be concrete and opinionated.

## Answer

# Open-Weight LLM Selection & Deployment on M5 Pro (64 GB) for a Financial Research Engine

## 1. The Hardware Constraint, Made Precise

Your 64 GB unified memory pool breaks down as:

| Component | Budget |
|---|---|
| macOS + window server + background apps | ~10–12 GB |
| **Available for model + KV cache + runtime** | **~52 GB** |
| Your target (conservative) | **45 GB** |
| Reserve for KV cache at long context | 6–12 GB |
| **Usable for weights** | **33–39 GB** |

Memory bandwidth on an M5 Pro will likely land in the 150–273 GB/s range (extrapolating from M4 Pro's 120 GB/s and the trajectory Apple has shown). At that bandwidth, your **tokens/s ceiling** for a dense model is roughly:

```
tokens/s ≈ bandwidth / (weight_bytes_per_token_read)
```

For a 32B model at Q8_0 (~35 GB), each token decode reads the full weight tensor:
- Best case: 150 GB/s ÷ 35 GB ≈ **4 tok/s**
- With 273 GB/s: ≈ **7–8 tok/s**

For a MoE model with 3–5B active params, you only read the active experts:
- 5B active at Q8_0 ≈ 10 GB → **15–27 tok/s**

This is the single most important constraint. It dictates architecture choice more than parameter count.

---

## 2. The Mid-2026 Open-Weight Landscape (Realistic Assessment)

### Models that fit in ≤ 45 GB

| Model (mid-2026 projection) | Architecture | Size at Q8_0 | Size at Q5_K_M | Context | Notes |
|---|---|---|---|---|---|
| **Qwen3.5-32B** (or Qwen4-32B) | Dense, GQA | ~35 GB | ~27 GB | 128K native | Strongest dense model in this class |
| **Qwen3.5-30B-A3B** | MoE (30B total, 3B active) | ~28 GB | ~21 GB | 128K | Fast inference, slightly lower ceiling |
| **Llama 5-34B** (projected) | Dense, GQA | ~36 GB | ~28 GB | 128K | If Meta ships a 34B successor to Llama 3 70B |
| **DeepSeek-R2-Distill-32B** (projected) | Dense | ~35 GB | ~27 GB | 64–128K | If they distill R2 reasoning into a smaller model |
| **Mistral Large 2-48B** (projected) | Dense | ~50 GB | ~38 GB | 128K | Might be tight at Q5, fine at Q4 |
| **Command A-34B** (Cohere) | Dense | ~36 GB | ~28 GB | 128K | Best-in-class tool calling |
| **Yi-34B / Yi-Lightning-Small** | Dense/MoE | ~30–36 GB | ~24–30 GB | 128K | Strong multilingual, good at finance |
| **GLM-5-32B** (Zhipu) | Dense | ~35 GB | ~27 GB | 128K | Strong Chinese+English, good at structured extraction |

### Models that DON'T fit (for reference)

| Model | Why it's out |
|---|---|
| DeepSeek V3/R1 (671B MoE) | ~400+ GB even at Q4 |
| Llama 4 Maverick (400B MoE) | ~250 GB at Q4 |
| Llama 4 Scout (109B MoE) | ~65 GB at Q4, ~50 GB at Q3 — too tight for KV |
| Qwen3-235B-A22B | ~130 GB at Q4 |
| Mixtral 8x22B (141B) | ~80 GB at Q4 |

---

## 3. Head-to-Head Comparison on Your Criteria

### Reasoning (financial analysis, risk assessment, causal inference)

| Model | Expected quality | Why |
|---|---|---|
| Qwen3.5-32B | ★★★★★ | Qwen3 already beat Llama 3 70B on MMLU, GSM8K, HumanEval. The mid-2026 iteration with RLHF + reasoning data should be very strong. |
| DeepSeek-R2-Distill-32B | ★★★★★ (reasoning-specific) | If DeepSeek distills their chain-of-thought RL into a 32B model, pure reasoning would be best-in-class. But general capability might trail Qwen. |
| Llama 5-34B | ★★★★ | Meta's RL + post-training has improved, but Qwen and DeepSeek have been ahead on reasoning benchmarks since late 2024. |
| Mistral Large 2-48B | ★★★★ | Strong but Mistral has been less aggressive on reasoning data than DeepSeek/Alibaba. |
| Command A-34B | ★★★★ | Good general reasoning, Cohere optimizes for "useful" rather than "deeply reasoning." |

### Coding (for your ETL scripts, any code generation in the engine)

| Model | Expected quality |
|---|---|
| Qwen3.5-32B | ★★★★★ (Qwen3 was already competitive with Claude 3.5 on HumanEval/MBPP) |
| Llama 5-34B | ★★★★ |
| DeepSeek-R2-Distill-32B | ★★★★ (reasoning helps with complex code) |
| Mistral Large 2-48B | ★★★★ |

### Long Context (reading 10-Ks, earnings transcripts, multi-article synthesis)

This is where architecture matters most:

| Model | Native context | Effective 128K quality | Notes |
|---|---|---|---|
| Qwen3.5-32B | 128K | ★★★★★ | Qwen3 showed strong needle-in-haystack and long-doc comprehension. Mid-2026 version should be better. |
| Llama 5-34B | 128K | ★★★★ | Meta's 128K is real but quality degrades past 64K somewhat. |
| Mistral Large 2-48B | 128K | ★★★★ | Mistral has been good at long context. |
| Command A-34B | 128K | ★★★★ | Cohere optimized for this. |
| DeepSeek-R2-Distill-32B | 64K (likely) | ★★★ | If the distilled model only supports 64K, that's a limitation for full 10-Ks. |

### Tool Use / Function Calling

| Model | Quality | Notes |
|---|---|---|
| Command A-34B | ★★★★★ | Cohere specifically engineered for this. Most reliable tool-calling in the open space. |
| Qwen3.5-32B | ★★★★☆ | Qwen3 added strong tool-calling. Mid-2026 should be very good. |
| Llama 5-34B | ★★★★ | Meta's tool-calling is solid but occasionally hallucinates tool args. |
| Mistral Large 2-48B | ★★★★ | Good. |
| DeepSeek-R2-Distill-32B | ★★★ | Reasoning-focused; tool-calling may be weaker unless specifically trained for it. |

---

## 4. My Pick: **Qwen3.5-32B** (or its direct successor) at **Q6_K** quantization

### Why this specific model + quant

1. **Best all-around for your use case.** Qwen3 was the strongest open model in the 30–35B class on reasoning, coding, long-context, AND tool use simultaneously. No other model in this size class is as balanced. By mid-2026, the successor will be further ahead.

2. **Q6_K is the sweet spot.** At ~31 GB for weights, you leave 14 GB for KV cache + runtime, which supports:
   - 128K context with Q8_0 KV cache (~14 GB) — just fits
   - OR 128K context with Q4_0 KV cache (~7 GB) — comfortable margin
   - OR 256K context with Q4_0 KV — possible but tight

   Q8_0 would be ideal quality-wise but leaves you only 10 GB for KV, forcing Q4_0 KV at 128K (quality loss on the cache hurts long-context accuracy). Q6_K gives you the best quality-to-margin ratio.

3. **Dense architecture = consistent quality.** For financial analysis where every token matters (you don't want a MoE routing error to drop a critical signal), a dense model is more predictable. MoE speed is nice for interactive use, but for overnight batch where you have 8–10 hours, dense quality > MoE speed.

4. **Ecosystem maturity.** Qwen models are the best-supported in MLX (Apple's native framework) and llama.cpp. You'll have day-one support with good Metal optimizations.

### When I'd deviate

- If you find yourself needing **maximum reasoning depth** (e.g., complex causal chains in macro analysis) and the Qwen model occasionally "loses the thread" on very long chains → swap in **DeepSeek-R2-Distill-32B** for that specific pipeline stage, keep Qwen for everything else.
- If **tool-calling reliability** becomes your top pain point (the engine calls 50+ tools per night) → keep **Command A-34B** as a dedicated tool-dispatch layer, with Qwen doing the heavy reasoning.

---

## 5. Hosting Stack on M5 Pro

### Primary Runtime: **MLX** (not llama.cpp, not Ollama)

Here's my reasoning:

| Factor | MLX | llama.cpp (Metal) | Ollama |
|---|---|---|---|
| Memory management | Native unified-memory aware, zero-copy between CPU/GPU | Good but copies in some paths | Wraps llama.cpp |
| Throughput on Apple Silicon | Best (designed for it) | 90–95% of MLX | 90–95% of MLX |
| KV cache quantization | Supported (q8, q4) | Supported (more options) | Limited |
| Speculative decoding | Basic support | Good (draft model) | No |
| API for serving | `mlx_lm.server` (OpenAI-compatible) | `llama-server` (OpenAI-compatible) | Built-in |
| Batch inference | Good | Good | Decent |
| Multi-model serving | Manual | Manual | Built-in |
| Fine-tuning / LoRA | Excellent (native) | Possible | No |

**Decision: MLX as the primary engine**, with `mlx_lm.server` exposing an OpenAI-compatible API. This gives you:
- Best tokens/s on your hardware
- Native unified memory management (no copies)
- OpenAI API compatibility so your research engine code doesn't change if you swap models
- Easy LoRA fine-tuning later (e.g., on your own financial data)

### Quantization Strategy (Layered)

```
Model weights:    Q6_K (MLX supports 6-bit; ~31 GB)
KV cache:        Q8_0 (MLX supports 8-bit KV; ~14 GB at 128K)
                   → Total: ~45 GB ✓
                  OR Q4_0 if you want 256K context headroom
```

If MLX's Q6_K support for the specific Qwen3.5-32B checkpoint is immature at release, fall back to:
- **Q8_0 weights + Q4_0 KV at 128K** (35 + 7 = 42 GB) — slightly lower weight quality but much better KV fidelity for long-context accuracy.

### Context Window Configuration

For your three pipeline stages:

| Stage | Context needed | Config |
|---|---|---|
| Overnight ETL (LLM-assisted extraction) | 32–64K per doc | `ctx=65536`, Q8_0 KV |
| Tripwire interpretation | 16–32K (5–10 articles + price data) | `ctx=32768`, Q8_0 KV |
| Morning digest narration | 8–16K (structured summary) | `ctx=16384`, Q8_0 KV |
| Interactive research (you, deep dive) | 128K+ | `ctx=131072`, Q4_0 KV |

You can reconfigure the server or spin up a second instance for the 128K interactive session. In practice, run the overnight pipeline with a 64K server, then restart for interactive use with 128K. Or, if you want both running simultaneously, use Q4_0 KV at 64K (7 GB) for the overnight worker and Q4_0 at 32K (3.5 GB) for interactive — total KV ~10 GB, leaving room.

### Speculative / Multi-Token Prediction

**Honest assessment: don't bother on M5 Pro for a dense 32B model.**

- Speculative decoding with a separate draft model (e.g., Qwen3-7B) would consume 5–8 GB of your already-tight RAM budget.
- The bandwidth bottleneck (reading 31 GB of weights per token) means even a perfect draft model only helps if acceptance rate is > 70%, and the memory cost of the draft model eats your KV budget.
- For **overnight batch** (your primary workload), you have 8–10 hours. At 5–7 tok/s, that's 144K–250K tokens of generation — plenty for your pipeline.
- For **interactive use**, the 5–7 tok/s is acceptable for a research assistant. You're not building a chatbot.

**Exception:** If Apple ships M5 Pro with a native MTP (multi-token prediction) head in the model architecture (like DeepSeek's MTP module), that would be free speed with no extra memory. Check if Qwen3.5 includes an MTP head — if so, enable it.

### Serving Stack Architecture

```
┌─────────────────────────────────────────────────────────┐
│  macOS (macOS 20.x, mid-2026)                          │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  mlx_lm.server (port 8080)                     │    │
│  │  Model: Qwen3.5-32B Q6_K                       │    │
│  │  KV: Q8_0, ctx=65536 (overnight)              │    │
│  │  OR ctx=131072, Q4_0 KV (interactive)         │    │
│  │  Exposes: /v1/chat/completions, /v1/completions│   │
│  └────────────────────┬────────────────────────────┘    │
│                       │                                  │
│  ┌────────────────────▼────────────────────────────┐    │
│  │  Research Engine (Python/TypeScript)           │    │
│  │  ┌───────────┐ ┌──────────┐ ┌──────────────┐  │    │
│  │  │  ETL      │ │  Tripwire│ │  Digest      │  │    │
│  │  │  Pipeline │ │  Engine  │ │  Narrator    │  │    │
│  │  └─────┬─────┘ └────┬─────┘ └──────┬───────┘  │    │
│  │        │             │               │          │    │
│  │  ┌─────▼─────────────▼───────────────▼──────┐  │    │
│  │  │  OpenAI-compatible client (point at 8080)│  │    │
│  │  └───────────────────────────────────────────┘  │    │
│  └─────────────────────────────────────────────────┘    │
│                                                         │
│  ┌─────────────────────────────────────────────────┐    │
│  │  Data Layer                                    │    │
│  │  • SQLite/PostgreSQL (prices, signals)         │    │
│  │  • ChromaDB/Qdrant (vector store for docs)    │    │
│  │  • Parquet files (raw ETL output)             │    │
│  └─────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────┘
```

---

## 6. Wiring the Research Engine

### Pipeline Architecture

```
22:00 ──── ETL Kickoff (deterministic)
   │
   ├──► Price ingestion (Polygon/Alpha Vantage API → Parquet → SQLite)
   ├──► News ingestion (RSS + API
