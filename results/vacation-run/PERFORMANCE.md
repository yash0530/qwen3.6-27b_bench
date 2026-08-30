# Qwen 3.8 27B (+ 35B) — performance by depth

All measured through real Claude Code sessions, MTP speculative decoding on, on M5 Pro /
64 GB. Decode figures are MTP-effective (what you actually wait through). GGUF = llama.cpp
b10621, medium thinking; MLX = mlx_vlm 0.6.17, xhigh thinking (it ignores the effort field).

## Decode speed (tok/s) vs context depth

| Depth | gguf4 | gguf5 | gguf6 | gguf8 | mlx4 | 35B A3B |
|---:|---:|---:|---:|---:|---:|---:|
| ~19K | 15.6 | 15.1 | 15.9 | **17.1** | **25.8** | ~67 |
| ~65K | 11.3 | 11.3 | 11.2 | 13.7 | 21.8 (@58K) | (stays fast) |
| ~110K | 8.0 | 9.2 | 9.6 | 10.8 | 13.2 → OOM | — |
| ~157K | 7.8 | 8.3 | 8.0 | 8.6 | ✗ | — |
| ~201K | 6.9 | 6.6 | 6.5 | 7.4 | ✗ | — |

Note: with MTP, the *bigger* GGUF quants decode as fast or faster at shallow depth
(gguf8 17.1 vs gguf5 15.1) because the higher-quality target gives the draft head higher
acceptance. All quants converge to ~7 tok/s at 200K. MLX (mlx4) decodes ~1.7× faster than
GGUF at shallow depth — its one real advantage — but can't reach depth.

## Cold prefill speed (tok/s) vs depth

| Depth | GGUF (any quant) | mlx4 | 35B A3B |
|---:|---:|---:|---:|
| ~19K (cold) | 256–296 | 462 | ~910 |
| ~58–65K | ~180 | 682 | — |
| ~110K | ~120 | ~300 (OOMing) | — |
| ~160K | ~90 | ✗ | — |
| ~200K | ~60 | ✗ | 214 (deep resume) |

Prefill is what makes deep context slow to *build*: adding ~46K of new content at 200K
depth takes ~40 min. Warm restore avoids re-paying this — a follow-up at 200K reprocessed
only 122 tokens — but the *decode* is still ~7 tok/s, so even warm deep turns cost minutes.

## Max usable context & memory

| Model | Max correct recall | Wired @ depth | Snappy ceiling | Failure mode |
|---|---:|---:|---:|---|
| **gguf5** | **201K** | 42.5 GB | ~80K | none hit (time only) |
| gguf4 | ~180–201K | 39.4 GB (most free) | ~80K | none hit (time only) |
| gguf6 | 156K | 46.6 GB | ~70K | time (slower prefill) |
| gguf8 | 156K | 50.1 GB (**edge**) | ~60K | time + memory edge |
| mlx4 | 58K (OOM 80K) | spikes to 56 GB | ~50K | **GPU OOM** at depth |
| 35B A3B | 262K native | 37 GB | large | none (fast MoE) |
| ~~mlx6~~ | ~~58K~~ | — | — | **kernel panic** (deleted) |
| ~~mlx8~~ | ~~41K~~ | — | — | **dead** at 41K (deleted) |

## Warm follow-up latency (the number you feel most)

| Depth | GGUF | mlx4 | 35B A3B |
|---:|---|---|---|
| shallow (<60K) | 8–33 s | **2–4 s** | **~2 s** |
| deep (~200K) | minutes (decode-bound) | ✗ | (untested deep) |

## One-line takeaways

- **Snappy work under ~80K context** → gguf5 (or mlx4 for the fastest shallow turns).
- **Deep context (100–200K)** → gguf5/gguf4 only; possible and correct, but slow
  (minutes/turn) — treat as batch, not interactive.
- **Max quality** → gguf8 (works to 156K, at the memory edge).
- **Fastest overall** → 35B A3B (67 tok/s, 262K native, 2 s warm) when its older
  generation's quality suffices.
- **The wall on every 27B config**: prefill *time* and deep-decode *speed*, not memory
  (GGUF) — the sweet spot is well under 100K on all of them.
