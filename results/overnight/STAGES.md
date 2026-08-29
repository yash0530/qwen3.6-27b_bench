# Findings & run log — overnight 2026-08-22

## Pre-run findings (CPU-side, no GPU used)

- **Blocker**: two active Claude Code sessions driving `llm-serve start 35b` (qwen3.6-35b,
  35 GB mlock'd) since 02:20:36. RAM pinned at 26% free ⇒ no second engine fits.
  idle_watch.sh polls; GPU stages start when it drains (20 min idle) — NOT killed.
- **Patch tool built & validated** (apply→py_compile→revert):
  `local-setup/scripts/apply-latest-only-patch` gates the two dead full-prompt stores
  (`dispatch.py` n==0 store, `ar.py` exact-mode harvest) behind APC_SKIP_FULL_STORE
  (default skip=1; =0 restores upstream). Checkpoint stores untouched.
- **Baseline numbers from REPORT.md** (decode t/s @ Qwen 3.8 27B):
  MLX-4bit 27.2 · MLX-6bit 21.1 · MLX-8bit 16.3 · GGUF best (UD-Q4_K_XL+MTP) 12.1 ·
  GGUF Q8_0 13.5. GGUF ≈ half of MLX ⇒ MLX is the speed path (user's instinct right).
- **Warm TTFT precedent**: 3.8 MLX-8bit cold 84.5s → warm 1.28s (98%); GGUF Q8 76.9 → 1.50s.
  llama.cpp caching demonstrably works on 3.8 hybrids in bench conditions.
- Live 35b session logs show "forcing full prompt re-processing ... hybrid/recurrent memory"
  on qwen3.6-35b — that arch loses llama.cpp cache warmth; 3.8 may differ. To verify on 3.8.
- GGUF downloaded: ~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-UD-Q4_K_XL.gguf (16 GB).
  Trial A will run on :8091 (8089 occupied).

## Planned GPU ladder (after RAM frees)
- A: llama.cpp :8091, -c 65536 --cache-reuse 256, MTP off then draft-mtp — stability/cache reference.
- B (Stage A config): MLX ports 8093/8791, LLM_APC=1 ENTRIES=2 SPEC=0 — 15 turns.
- C (Stage B patch): same + checkpoint-only patch — 15 turns.
- D (GOAL): C + LLM_SPEC=1 (MTP drafter) — 15 turns ← target config.
[03:25:25] stopped idle 35b stack (user mandate; restart: llm-serve start 35b)

## 2026-08-22 03:25 — Trial A launch
- config: llama-server :8091, UD-Q4_K_XL, -c 65536, --cache-reuse 256, MTP off
- uptime:  3:25  up 5 days, 12:33, 4 users, load averages: 1.23 0.98 0.93

## 2026-08-22 03:28 — Trial A: 15 turns direct-attach
- config: llama.cpp :8091 UD-Q4_K_XL -c65536 cache-reuse 256 MTP off
- uptime:  3:28  up 5 days, 12:36, 4 users, load averages: 1.22 1.07 0.96

## 2026-08-22 03:36 — Trial A v2: 15 turns via proxy
- config: claude → llm-proxy :8093 (system-hoist+stripVolatile) → llama.cpp :8091, MTP off
- uptime:  3:36  up 5 days, 12:44, 4 users, load averages: 0.77 1.14 1.06

## Trial A result (03:28–04:00) — REJECTED
- Native /v1/messages works (thinking blocks, tools via proxy). BUT: Claude Code sends a second
  role:"system" message inside messages[] which this model's strict Jinja template rejects
  ("System message must be at the beginning") — llm-proxy.mjs's system-hoist fixes it.
- **Warmth: 29 requests, 0 warm hits** — every request full-prompt reprocesses
  ("lack of cache data ... SWA or hybrid/recurrent memory"). Verified prompts were clean
  91.8% extensions ⇒ engine-side limitation, not our rendering. Recurrent GDN state can't
  fork mid-sequence like KV.
- Decode 13.3 t/s (≈½ of MLX-4bit's 27.2). Prefill ~290 t/s ⇒ ~90 s per cold prefill @25k.
- Turns: t1=900s(capped) t2=550s. No crash; Metal DeltaNet healthy.
- Verdict: unusable for the speedup goal. MLX ladder proceeds as planned.

## 2026-08-22 04:05 — MLX Stage A: 15 turns
- config: mlx4, APC=1 ENTRIES=2, MTP off, unpatched (upstream default entries)
- uptime:  4:05  up 5 days, 13:13, 4 users, load averages: 1.14 0.94 0.89

## 2026-08-22 04:36 — MLX Stage B: checkpoint-only patch
- config: mlx4, APC=1 ENTRIES=2, MTP off, APC_SKIP_FULL_STORE=1 (patched)
- uptime:  4:36  up 5 days, 13:45, 4 users, load averages: 1.25 1.63 1.56

## 2026-08-22 04:37 — MLX Stage B: 15 turns
- config: mlx4, ENTRIES=2, MTP off, checkpoint-only patch active
- uptime:  4:37  up 5 days, 13:45, 4 users, load averages: 1.25 1.59 1.55

## Stage B interim observations (04:35–)
- Patch verified active (dispatch.py + ar.py "patched").
- Cold-prefill pattern: full 60-75s re-prefills every ~5th request, gaps almost constant
  (~1174/1187/1232/1253/1161 tok) ⇒ systematic early-prompt mutation every ~1200 tokens,
  NOT random divergence. Suspect a second accumulating block in Claude Code's system prompt
  that stripVolatile doesn't cover (v2.1.239), or lazy tool-list growth. Diagnostic with
  DUMP_DIR queued after this run: diff last-warm vs cold request → locate volatile bytes.
- Note: server.log truncates on llm-serve restart — archived to server-stageb.log.

## Stage B result (04:35–05:40) — PASSED
- 15/15 turns rc=0, no panics, free ≥37% throughout, ended clean.
- Requests: 43 warm / 12 cold. Warm prefills ~13-29k tok/s; colds 426-436 tok/s (60-75 s).
- CORRECTION to plan wording: checkpoint ≈ full-prompt size (n−16). Patch does NOT halve
  snapshot BYTES; it halves per-cold-prefill big-copy events (2→1) — the churn implicated
  in the panics — and lets entries=N hold N distinct states instead of N/2.
- Open question: period-5 colds with near-constant ~1200-token gaps ⇒ systematic early-prompt
  mutation. Diagnostic next.

## 2026-08-22 05:42 — Diagnostic: dump-enabled 6 turns
- config: same patched server, DUMP_DIR on :8791 proxy, catch a cold transition
- uptime:  5:42  up 5 days, 14:50, 4 users, load averages: 1.32 1.71 1.82

## 2026-08-22 05:58 — Diagnostic 2: stripVolatile v2
- config: task-tools nudge stripped; expect near-zero mid-run colds
- uptime:  5:58  up 5 days, 15:06, 4 users, load averages: 1.64 1.42 1.56

## 2026-08-22 06:24 — Stage D: GOAL CONFIG
- config: mlx4 + checkpoint-only patch + ENTRIES=2 + MTP block=3 + stripVolatile v2
- uptime:  6:24  up 5 days, 15:32, 4 users, load averages: 1.34 1.54 1.59

## 2026-08-22 06:41 — Stage K: GOAL+kv8
- config: mlx4 + patch + ENTRIES=2 + MTP + --kv-bits 8 (manual launch)
- uptime:  6:41  up 5 days, 15:49, 4 users, load averages: 1.35 1.69 1.70

## Stage D final (06:23–07:00) — GOAL CONFIG PROVEN
- 14 turns rc=0 across three segments; 3 rail trips (25%/22%/14% free) all correct, all recovered at idle.
- Zero panics all night. Drafter loaded ("speculative decoding enabled").
- Walls: median 41s, min 9s (turns of 9/9/12s = MTP at work) vs Stage B median 134s min 42s.
- Server steady RSS 15.9 GB; transient metal-heap growth ~15-20GB during active turns.

## Stage K (06:38–07:00) — kv-bits 8 REJECTED
- Streaming tool-call responses truncate: stream ends with tool_calls deltas but NO
  finish_reason chunk (4 lines total). Non-streaming well-formed. Claude Code ends turn
  silently on truncated streams → --continue sessions no-op. Upstream bug in mlx_vlm
  0.6.15 kv-quant + streaming path. Manual launch otherwise equivalent to llm-serve's.

## Night close (07:00)
All engines stopped, 90% free, panic count 4 (baseline), uptime continuous.
