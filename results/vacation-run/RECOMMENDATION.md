# Qwen 3.8 27B — final recommendation (vacation campaign, 2026-08-28 → 08-30)

All numbers measured through real Claude Code sessions on M5 Pro / 64 GB. Full data:
STRESS.md (MLX), STRESS-GGUF.md + gguf-turns/ (GGUF), QUALITY.md, SNAPSHOT_RESEARCH.md.

## The headline

**The right daily driver is now `gguf5` (Qwen3.8-27B-UD-Q5_K_XL on llama.cpp), not `mlx4`.**
The August llama.cpp upgrade (b9620 → b10621) fixed hybrid-model warm caching, which flips
the entire engine choice: GGUF now holds a warm cache under Claude Code, with a flat memory
profile that reaches 2–3× deeper than MLX and makes 5/6/8-bit actually usable — recovering
the quality tier MLX could never run safely on this machine.

## Cross-engine results

| Config | Engine | Warm follow-up (shallow) | Deepest correct turn | Wired @ depth | Decode t/s | Quality | Safe? |
|---|---|---|---:|---:|---:|---:|:--:|
| **gguf5** Q5_K_XL | llama.cpp | 8–33 s | **201K** | 42.5 GB | 15.8→7.8 (by depth) | high | ✅ flat mem |
| gguf4 UD-Q4_K_XL | llama.cpp | 8–17 s | ~180K | 39.4 GB (most headroom) | fastest GGUF | good | ✅ |
| gguf6 UD-Q6_K_XL | llama.cpp | ~s | 156K | 46.6 GB (tight) | slower | higher | ✅ |
| gguf8 Q8_0 | llama.cpp | ~s | 156K | 50.1 GB (**edge**) | slowest | highest | ⚠️ mem edge, no panic |
| mlx4 | mlx_vlm | **2–4 s** | 58K (OOM 80K) | spikes to 56 GB | **21–26** | ~= gguf5 | ⚠️ OOM at depth |
| mlx6 | mlx_vlm | — | 58K | — | 9–12 | higher | ❌ kernel-panicked |
| mlx8 | mlx_vlm | — | **41K (dead)** | — | ~10 | highest | ❌ unusable |
| 35B A3B Q8 | llama.cpp | **2 s** | 262K native | 37 GB | **67** | (older gen) | ✅ |

## Why gguf5

1. **Warm cache works and holds deep.** 12/12 warm at shallow depth; the cache-ram fix
   (below) proven to keep a 201K context warm — the follow-up prefilled only 122 new
   tokens instead of re-processing 201K.
2. **Comfortable memory.** Flat 42.5 GB wired with headroom to spare; no OOM, no panic,
   ever. (gguf8 works too but sits at the 50 GB edge; gguf4 has the most headroom but
   lowest quality.)
3. **Honours the 5/6/8-bit quality preference** MLX forced us to abandon.
4. **MTP speculative decoding active** (inline nextn head, 61–98 % draft acceptance).

## What each alternative is for

- **`mlx4` — snappy shallow work.** Fastest decode (21–26 t/s), 2–4 s trivial turns.
  Best when you stay under ~50K context and want minimum latency. Loses at depth (OOMs).
- **`gguf8` — maximum quality**, if you accept the slowest decode and running at the
  memory edge. Reaches 156K. This is the "8-bit is finally attainable" option (mlx8 was
  dead at 41K).
- **`35b` (Qwen 3.6 35B A3B) — fastest overall.** 2 s follow-ups, 67 t/s, 262K native.
  Pick it when speed beats the 27B's newer-generation quality.

## Hard limits that apply to every config

- **Depth is bounded by prefill *time*, not memory or cache.** Adding ~46K of new content
  at ~200K depth takes ~40 min (prefill ~64 t/s at depth). Even a *warm* follow-up at
  200K costs minutes because deep decode is ~7 t/s. **The snappy sweet spot is < ~80K.**
- **Claude Code hard-rejects a single prompt over ~200K estimated tokens** ("Prompt is
  too long"), so deep context can only be built incrementally, never ingested in one shot.
- **kv-bits** stays off (upstream streaming bug); **`--cache-reuse`** is inert on this
  hybrid arch (removed).

## Config changes shipped (committed)

- `llm-serve`: `--cache-ram 24576` (was llama.cpp's 8192 default, which silently capped
  warm restore at ~79K); dropped inert `--cache-reuse`; added `gguf4/5/6/8`, `optiq4`,
  `mxfp8` aliases. `LLM_CACHE_RAM_MIB` overrides.
- MLX venv upgraded to mlx-vlm 0.6.17 + local `apply-latest-only-patch` and
  `apply-single-clone-patch` (single-clone halves the per-store GPU transient).
- `qwen-code`: `CLAUDE_CODE_MAX_CONTEXT_TOKENS` default 65536.

## Open / not done

- **`optiq4` (mixed 4-bit, GDN layers kept 8-bit) and `mxfp8` (M5 native FP8)** were
  researched and downloaded but not cleanly benchmarked (harness bugs, then MLX
  deprioritised once GGUF won). Worth a quick shallow test only if you want a
  higher-quality *snappy* MLX option than mlx4 — they will still OOM at depth like all MLX.
- **Thinking-level confound**: MLX sessions ran at `xhigh` (mlx_vlm ignores the effort
  field), GGUF at `medium`. A matched-effort rerun would tighten the shallow-latency
  comparison but wouldn't change the depth/safety verdict.
