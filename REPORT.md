# Local LLM Benchmarks — Multi-Model Analysis

_Speed sweep: 400 runs · full-length: 30 runs · temp 0.6, seed 42, ctx 16384._

## TL;DR

- **Qwen 3.6 27B (Q5)** — peak **17.7 tok/s** at draft-n=3 (1.37x vs off) (57% accept); quality 8.3/10; ~6795 tok/answer.
- **Qwen 3.6 27B (Q6)** — peak **17.5 tok/s** at draft-n=3 (1.61x vs off) (57% accept); quality 8.5/10; ~6781 tok/answer.
- **Qwen 3.6 27B (Q8)** — peak **14.7 tok/s** at draft-n=3 (1.67x vs off) (60% accept); quality 8.6/10; ~6703 tok/answer.
- **Qwen 3.6 35B A3B (Q5)** — peak **76.2 tok/s** at draft-n=2 (1.27x vs off) (67% accept); quality ungraded; ~7612 tok/answer.
- **Qwen 3.6 35B A3B (Q6)** — peak **70.5 tok/s** at draft-n=1 (1.24x vs off) (78% accept); quality ungraded; ~6782 tok/answer.
- **Qwen 3.6 35B A3B (Q8)** — peak **51.6 tok/s** at draft-n=1 (1.16x vs off) (79% accept); quality 7.2/10; ~7780 tok/answer.
- **Qwen 3.6 27B (MLX-8bit)** (MLX) — **15.8 tok/s**, quality 7.5/10, ~7037 tok/answer.
- **Qwen 3.6 35B A3B (MLX-8bit)** (MLX) — **70.7 tok/s**, quality 8.4/10, ~7180 tok/answer.

## Charts

![01_decode_tok_s.png](results/charts/01_decode_tok_s.png)

![02_speedup.png](results/charts/02_speedup.png)

![03_acceptance.png](results/charts/03_acceptance.png)

![04_prompt_speed.png](results/charts/04_prompt_speed.png)

![05_ttft.png](results/charts/05_ttft.png)

![06_tokens.png](results/charts/06_tokens.png)

![07_quality.png](results/charts/07_quality.png)

![08_quality_vs_speed.png](results/charts/08_quality_vs_speed.png)

![09_runtime_tok_s.png](results/charts/09_runtime_tok_s.png)

## Speed sweep (mean over questions)

| model/quant | draft-n | tok/s | ±sd | accept % | prompt tok/s | TTFT ms |
|---|---|---|---|---|---|---|
| Qwen 3.6 27B (Q5) | off | 12.9 | 0.2 | - | 257 | 814 |
| Qwen 3.6 27B (Q5) | mtp1 | 14.9 | 0.3 | 80 | 247 | 849 |
| Qwen 3.6 27B (Q5) | mtp2 | 13.4 | 0.6 | 67 | 246 | 853 |
| Qwen 3.6 27B (Q5) | mtp3 | 17.7 | 1.0 | 57 | 242 | 864 |
| Qwen 3.6 27B (Q5) | mtp4 | 17.2 | 1.2 | 49 | 241 | 868 |
| Qwen 3.6 27B (Q6) | off | 10.8 | 0.0 | - | 275 | 760 |
| Qwen 3.6 27B (Q6) | mtp1 | 15.3 | 0.1 | 79 | 266 | 782 |
| Qwen 3.6 27B (Q6) | mtp2 | 14.1 | 0.2 | 68 | 251 | 829 |
| Qwen 3.6 27B (Q6) | mtp3 | 17.5 | 0.6 | 57 | 244 | 856 |
| Qwen 3.6 27B (Q6) | mtp4 | 16.8 | 0.7 | 48 | 246 | 850 |
| Qwen 3.6 27B (Q8) | off | 8.8 | 0.9 | - | 248 | 151701 |
| Qwen 3.6 27B (Q8) | mtp1 | 13.2 | 2.2 | 81 | 241 | 156802 |
| Qwen 3.6 27B (Q8) | mtp2 | 14.7 | 2.9 | 70 | 236 | 158227 |
| Qwen 3.6 27B (Q8) | mtp3 | 14.7 | 3.1 | 60 | 231 | 157944 |
| Qwen 3.6 27B (Q8) | mtp4 | 13.9 | 3.1 | 52 | 230 | 158322 |
| Qwen 3.6 35B A3B (Q5) | off | 60.0 | 0.3 | - | 875 | 271 |
| Qwen 3.6 35B A3B (Q5) | mtp1 | 76.0 | 1.0 | 77 | 906 | 236 |
| Qwen 3.6 35B A3B (Q5) | mtp2 | 76.2 | 3.7 | 67 | 943 | 224 |
| Qwen 3.6 35B A3B (Q5) | mtp3 | 68.5 | 5.3 | 56 | 904 | 233 |
| Qwen 3.6 35B A3B (Q5) | mtp4 | 63.6 | 5.3 | 49 | 898 | 234 |
| Qwen 3.6 35B A3B (Q6) | off | 57.1 | 0.5 | - | 950 | 225 |
| Qwen 3.6 35B A3B (Q6) | mtp1 | 70.5 | 1.6 | 78 | 935 | 226 |
| Qwen 3.6 35B A3B (Q6) | mtp2 | 69.2 | 2.1 | 65 | 925 | 231 |
| Qwen 3.6 35B A3B (Q6) | mtp3 | 65.1 | 2.5 | 56 | 914 | 233 |
| Qwen 3.6 35B A3B (Q6) | mtp4 | 59.4 | 2.7 | 47 | 903 | 235 |
| Qwen 3.6 35B A3B (Q8) | off | 44.6 | 8.5 | - | 792 | 51591 |
| Qwen 3.6 35B A3B (Q8) | mtp1 | 51.6 | 12.5 | 79 | 752 | 55475 |
| Qwen 3.6 35B A3B (Q8) | mtp2 | 50.3 | 13.2 | 67 | 753 | 55781 |
| Qwen 3.6 35B A3B (Q8) | mtp3 | 47.1 | 13.0 | 58 | 753 | 55861 |
| Qwen 3.6 35B A3B (Q8) | mtp4 | 42.5 | 11.8 | 49 | 745 | 55997 |

## MLX by prompt depth and KV precision

_Decode rate at ~200 tok (`shallow`) does not predict the agent loop, which sends ~23k tokens before the user types. `kv` is the KV-cache precision: `fp16` unquantized, `8` matching llama.cpp's `-ctk q8_0 -ctv q8_0`._

| model/quant | tier | kv | config | tok/s | ±sd | accept % | prefill tok/s | TTFT ms | peak GB |
|---|---|---|---|---|---|---|---|---|---|
| Qwen 3.6 27B (MLX-8bit) | agent | 8 | off | 9.4 | 0.0 | - | 360 | 64110 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | 8 | mtp2 | 12.7 | 0.1 | 78 | 359 | 64360 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | 8 | mtp3 | 13.3 | 0.4 | 64 | 356 | 64777 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | 8 | mtp4 | 12.4 | 1.1 | 54 | 355 | 65043 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | 8 | mtp5 | 11.0 | 0.6 | 44 | 354 | 65371 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | fp16 | off | 9.4 | 0.0 | - | 360 | 64231 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | fp16 | mtp2 | 12.8 | 0.3 | 79 | 360 | 64111 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | fp16 | mtp3 | 13.4 | 0.3 | 64 | 357 | 64619 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | fp16 | mtp4 | 12.2 | 0.4 | 52 | 354 | 65295 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | agent | fp16 | mtp5 | 10.7 | 0.4 | 42 | 355 | 65094 | 41.3 |
| Qwen 3.6 27B (MLX-8bit) | deep | 8 | off | 8.4 | 0.0 | - | 288 | 222384 | 48.5 |
| Qwen 3.6 27B (MLX-8bit) | deep | 8 | mtp3 | 11.3 | 0.4 | 66 | 287 | 223164 | 48.5 |
| Qwen 3.6 27B (MLX-8bit) | deep | fp16 | off | 8.5 | 0.0 | - | 290 | 221237 | 47.7 |
| Qwen 3.6 27B (MLX-8bit) | deep | fp16 | mtp3 | 11.3 | 0.4 | 66 | 289 | 221992 | 48.5 |
| Qwen 3.6 27B (MLX-8bit) | shallow | 8 | off | 10.0 | 0.0 | - | 293 | 903 | 36.6 |
| Qwen 3.6 27B (MLX-8bit) | shallow | 8 | mtp2 | 14.6 | 0.2 | 81 | 343 | 726 | 36.6 |
| Qwen 3.6 27B (MLX-8bit) | shallow | 8 | mtp3 | 15.6 | 0.7 | 69 | 303 | 840 | 36.6 |
| Qwen 3.6 27B (MLX-8bit) | shallow | 8 | mtp4 | 14.7 | 0.4 | 57 | 299 | 890 | 36.6 |
| Qwen 3.6 27B (MLX-8bit) | shallow | 8 | mtp5 | 12.9 | 0.8 | 46 | 301 | 926 | 36.6 |
| Qwen 3.6 27B (MLX-8bit) | shallow | fp16 | off | 10.0 | 0.0 | - | 289 | 1214 | 35.3 |
| Qwen 3.6 27B (MLX-8bit) | shallow | fp16 | mtp2 | 14.8 | 0.3 | 83 | 338 | 743 | 35.7 |
| Qwen 3.6 27B (MLX-8bit) | shallow | fp16 | mtp3 | 15.8 | 0.7 | 70 | 309 | 832 | 36.0 |
| Qwen 3.6 27B (MLX-8bit) | shallow | fp16 | mtp4 | 15.1 | 1.1 | 60 | 302 | 881 | 36.3 |
| Qwen 3.6 27B (MLX-8bit) | shallow | fp16 | mtp5 | 13.3 | 0.8 | 48 | 303 | 921 | 36.6 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | 8 | off | 56.0 | 0.0 | - | 1646 | 14030 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | 8 | mtp2 | 59.1 | 1.3 | 78 | 1635 | 14130 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | 8 | mtp3 | 55.1 | 2.3 | 63 | 1633 | 14156 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | 8 | mtp4 | 49.1 | 2.6 | 51 | 1634 | 14164 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | 8 | mtp5 | 43.2 | 2.8 | 42 | 1635 | 14162 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | fp16 | off | 56.5 | 0.0 | - | 1649 | 14010 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | fp16 | mtp2 | 59.2 | 1.3 | 78 | 1638 | 14107 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | fp16 | mtp3 | 55.1 | 2.2 | 63 | 1634 | 14145 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | fp16 | mtp4 | 49.2 | 2.6 | 51 | 1636 | 14147 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | agent | fp16 | mtp5 | 43.2 | 2.8 | 42 | 1634 | 14170 | 43.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | deep | 8 | off | 44.9 | 0.0 | - | 1158 | 55395 | 47.7 |
| Qwen 3.6 35B A3B (MLX-8bit) | deep | 8 | mtp2 | 44.8 | 1.3 | 76 | 1158 | 55355 | 47.7 |
| Qwen 3.6 35B A3B (MLX-8bit) | deep | fp16 | off | 44.9 | 0.0 | - | 1162 | 55600 | 47.5 |
| Qwen 3.6 35B A3B (MLX-8bit) | deep | fp16 | mtp2 | 44.9 | 1.4 | 76 | 1161 | 55247 | 47.7 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | 8 | off | 64.6 | 0.0 | - | 807 | 286 | 41.1 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | 8 | mtp2 | 71.3 | 1.1 | 80 | 841 | 278 | 41.1 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | 8 | mtp3 | 69.3 | 2.6 | 65 | 846 | 286 | 41.1 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | 8 | mtp4 | 64.1 | 3.2 | 54 | 845 | 296 | 41.1 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | 8 | mtp5 | 57.1 | 3.7 | 46 | 822 | 314 | 41.1 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | fp16 | off | 64.5 | 0.1 | - | 659 | 921 | 40.6 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | fp16 | mtp2 | 70.7 | 1.2 | 80 | 844 | 350 | 40.7 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | fp16 | mtp3 | 69.2 | 2.6 | 65 | 846 | 291 | 40.9 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | fp16 | mtp4 | 64.0 | 3.2 | 54 | 845 | 302 | 41.0 |
| Qwen 3.6 35B A3B (MLX-8bit) | shallow | fp16 | mtp5 | 58.1 | 3.4 | 46 | 845 | 310 | 41.1 |

## Full-length output (8192 cap, off config)

| model/quant | total tok | thinking tok | answer tok | answers completed |
|---|---|---|---|---|
| Qwen 3.6 27B (Q5) | 6795 | 4413 | 2378 | 5/5 |
| Qwen 3.6 27B (Q6) | 6781 | 4397 | 2379 | 5/5 |
| Qwen 3.6 27B (Q8) | 6703 | 4349 | 2350 | 5/5 |
| Qwen 3.6 35B A3B (Q5) | 7612 | 5517 | 2091 | 5/5 |
| Qwen 3.6 35B A3B (Q6) | 6782 | 4267 | 2511 | 5/5 |
| Qwen 3.6 35B A3B (Q8) | 7780 | 5029 | 2747 | 5/5 |
| Qwen 3.6 27B (MLX-8bit) | 7037 | 4647 | 2386 | 5/5 |
| Qwen 3.6 35B A3B (MLX-8bit) | 7180 | 4546 | 2628 | 5/5 |

## Output determinism (MTP correctness probe, #23302)

- llama.cpp: 212/320 diverged · MLX: 104/180 diverged

- 316/500 MTP runs produced a different output than their MTP-off baseline (fixed seed). 0 = MTP is output-preserving on this build; >0 flags the known determinism bug.

## Warm-cache TTFT (multi-turn session at agent depth)

_The deciding measurement. An agent loop re-sends a large stable preamble every turn, so what matters is TTFT on turns 2+, not the cold first turn._

| model | engine | cold (t1) | warm mean (t2-t5) | reduction |
|---|---|---|---|---|
| Qwen 3.6 27B (Q8) | gguf | 85.0 s | **1.68 s** | 98% |
| Qwen 3.6 27B (MLX-8bit) | mlx | 55.4 s | **55.95 s** | -1% |
| Qwen 3.6 35B A3B (Q8) | gguf | 24.9 s | **0.57 s** | 98% |
| Qwen 3.6 35B A3B (MLX-8bit) | mlx | 14.1 s | **12.74 s** | 10% |

## Append-only continuation (MLX)

_mlx_vlm.server re-renders the chat template each request, which forces a cache rewind Qwen 3.6's hybrid cache cannot do. Appending to the token sequence the cache already holds needs no rewind — so MLX's warm-cache deficit is a serving-layer gap, not an engine limitation._

| model | cold TTFT | warm TTFT | speedup | output preserved |
|---|---|---|---|---|
| Qwen 3.6 27B (MLX-8bit) | 55.83 s | **0.49 s** | 115x | NO (see notes) |
| Qwen 3.6 35B A3B (MLX-8bit) | 12.63 s | **0.14 s** | 87x | yes |

## Concurrency (shallow prompts, aggregate chars/s | mean TTFT)

_`gguf-mtp` is `-np 1` + MTP (requests queue; what llm-serve runs). `gguf-batch` is `-np N` with MTP off. llama.cpp cannot do both._

**qwen3.6-27b**

| arm | c=1 | c=2 | c=3 | c=4 | c=6 | c=8 | c=10 |
|---|---|---|---|---|---|---|---|
| mlx | 52 · 0.6s | 30 · 1.4s | 32 · 1.7s | 31 · 2.7s | 31 · 3.7s | 31 · 4.7s | 32 · 106.1s |
| gguf-mtp | 62 · 1.0s | 69 · 14.8s | 70 · 30.4s | 69 · 45.9s | 68 · 75.4s | 69 · 105.5s | 70 · 137.8s |
| gguf-batch | 34 · 1.0s | 66 · 1.6s | 90 · 2.5s | 106 · 3.3s | 113 · 4.9s | 120 · 6.0s | 109 · 7.7s |

**qwen3.6-35b-a3b**

| arm | c=1 | c=2 | c=3 | c=4 | c=6 | c=8 | c=10 |
|---|---|---|---|---|---|---|---|
| mlx | 254 · 0.3s | 254 · 0.5s | 272 · 0.7s | 277 · 1.0s | 284 · 1.5s | 301 · 1.9s | 311 · 12.2s |
| gguf-mtp | 235 · 0.3s | 263 · 4.0s | 274 · 7.8s | 281 · 11.3s | 277 · 18.7s | 281 · 26.0s | 283 · 33.6s |
| gguf-batch | 198 · 0.3s | 318 · 0.5s | 363 · 0.6s | 410 · 0.8s | 420 · 1.3s | 451 · 1.5s | 411 · 2.0s |

## Machine drift control

- Re-measured an unchanged GGUF cell (tier `agent`, draft-n 3): decode **-4.5%**, prefill **-5.7%** versus the stored value. Cross-engine margins below ~5% are unresolved.

## Measurement notes and known limitations

- **Arm parity is gated by `validate_parity.py`.** Run it before trusting any GGUF-vs-MLX number here. It exists because an earlier revision compared the two arms for ~250 runs while they were doing *different work*: `enable_thinking` was passed to `stream_generate` but not to `apply_chat_template`, so the Qwen3.6 template emitted a pre-closed `<think></think>` block and the MLX arm never reasoned, while llama.cpp under `--jinja` reasoned throughout. Both arms ran green and produced plausible tok/s. The checker measures reasoning rate per arm (95% vs 0% at the time) and fails on a gap over 15%, along with prompt-length parity, empty answers, missing acceptance, and cache leakage.
- **Acted on, and verified end-to-end.** The fp16-KV finding below was applied to `local-setup` (`llm-serve`, branch `tune/kv-fp16-and-draft-depth`) and re-measured on the live server with the 35B at 23k depth: RSS 38.7 -> 41.0 GB (50 GB wired limit), cold TTFT 28.90 -> 24.01 s, decode +11.6%, warm turn 5.27 -> 4.76 s. The 27B draft depth was raised 2 -> 3 in the same change.
- **MLX does not reuse prompt prefixes for Qwen 3.6; llama.cpp does.** This is the single most decision-relevant finding, because an agent loop re-sends a large stable preamble every turn. Measured over a 5-turn session at 23k depth: llama.cpp warm TTFT **1.68 s** (27B) / **0.57 s** (35B) against cold 85 s / 24.9 s, i.e. near-perfect reuse. MLX stays flat at ~55.5-56.6 s (27B) and ~12.6 s (35B) across all turns.
  - mlx_vlm's Automatic Prompt Caching is **off unless `APC_ENABLED=1`** (`apc.py:3769`). Enabling it — verified live via `/v1/cache/stats` — changed nothing: `lookups_hit=0, lookups_miss=0, stores=0`.
  - Probable cause is architectural. Qwen 3.6 is hybrid: `qwen3_5/language.py:2615` returns `ArraysCache` for Gated Delta Network layers and `KVCache` for attention layers. Block-mode APC requires *every* entry to be a `KVCache` (`apc.py:292`), so it cannot apply. The 'exact' whole-prefix fallback nominally accepts `ArraysCache` but stores nothing in practice. llama.cpp's slot cache does longest-common-prefix reuse, which survives a growing conversation; exact whole-prefix snapshots do not.
- **MLX's continuous batching underperforms under load.** At 10 concurrent clients llama.cpp `-np N` (MTP off) beats MLX on aggregate throughput, mean TTFT and max latency for both models; on the 27B MLX's max latency is 644 s vs 193 s. So while mlx_vlm *can* batch and speculate simultaneously — which llama.cpp cannot — the combination does not pay off here.
- **Machine drift over the measurement window was -4.5% decode / -5.7% prefill** (`drift_check.py`, re-measuring an unchanged GGUF cell). Cross-engine margins below ~5% should be treated as unresolved.
- **Quantized KV + MTP is broken in mlx_vlm 0.6.3 at depth.** `models/qwen3_5/language.py:1481` does `keys.shape[-2]` on the KV cache, but a quantized MLX cache is a *list* of (values, scales, biases), so the speculative-verify path raises `AttributeError: 'list' object has no attribute 'shape'`. Measured: shallow+q8+MTP passes, agent+fp16+MTP passes, agent+q8+no-MTP passes, agent+q8+MTP fails at every block size. `turboquant` fails differently.
- Consequently the MLX q8 arm is labelled **q8 decode-only**: quantization is deferred past the prompt (`quantized_kv_start`), so the prompt's KV stays unquantized. That runs, but it does **not** deliver q8 KV's memory saving at long context and is therefore *not* equivalent to llama.cpp's `-ctk q8_0 -ctv q8_0`. There is currently no MLX equivalent of the llm-serve config.
- **MLX draft-block-size 1 is a silent no-op**: the drafter proposes `block_size - 1` tokens, so 1 proposes none, emits a single token and stops — reporting an absurd tok/s. Valid speculative depths start at 2.
- `prefill_step_size` is pinned at 2048. Raising it above the prompt length to avoid chunked prefill OOMs Metal at 23k tokens.
- Legacy `mlx_lm.server` records are excluded from this report: they had no speculative decoding, a corrupted thinking/answer split, and TTFT-derived prompt speed. They remain in results.jsonl for provenance only.
- Prompt text is byte-identical across runtimes (built once, cached to `results/prompt_tiers.json`), so both engines tokenize the same bytes at each depth tier rather than being compared on prompts that differ by a few tokens.

## Hosting recommendation

On this machine, serve **Qwen 3.6 27B (Q8)** with peak speed. Launch line:

```bash
llama-server -m /Users/yash/Models/qwen3.6-27b-mtp-q8/Qwen3.6-27B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 3 \
  -c 16384 -ngl 99 -fa on -np 1 --jinja --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```
