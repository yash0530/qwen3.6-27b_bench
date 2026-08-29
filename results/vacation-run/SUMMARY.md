# Vacation-run summary — Qwen 3.8 27B best-quant campaign
_2026-08-28 18:45 → 2026-08-29 ~02:00, M5 Pro / 64 GB. Detail: STAGES.md, quality/QUALITY.md._

## Adopted config (already the default — `llm-serve start mlx4` is all you run)

**mlx4 + MTP block=3 + APC entries=2 + checkpoint-only patch, on mlx-vlm 0.6.17
(`local_llm_bench/.mlxenv-0617`), with `CLAUDE_CODE_MAX_CONTEXT_TOKENS=65536` in qwen-code.**

Measured through real Claude Code sessions on the adopted config:
- trivial warm turn **4 s** (target was 4–5 s), substantive warm turns 37–82 s
- warm prefill 0.9–9 s at 20–28k tokens (vs 45–60 s cold), 97% warm-hit rate
- decode ~21–22 tok/s, zero panics / zero Metal OOMs on this config

## Why not 8-bit or 6-bit (the original preference)

1. **Safety**: mlx6 + APC kernel-panicked the machine at 22:47
   (`"completeMemory() prepare count underflow" @IOGPUMemory.cpp:550` — same driver
   assert as the Aug 15/16 panics). Trigger: APC's multi-GB snapshot clone near the
   GPU wired limit. mlx4 has enough slack to fail gracefully; mlx6/mlx8 do not.
   Warm caching on 6/8-bit is permanently off this machine.
2. **Speed**: decode 17.6 (mlx4) vs 12.4 (mlx6) vs 9.7 (mlx8) tok/s; RSS 15.5 vs
   21.7 vs 25.8 GB.
3. **Quality**: judged means 7.78 (mlx4) / 6.80 (mlx6) / 8.27 (mlx8) at n=1 —
   non-monotonic, i.e. sampling variance exceeds the quantization effect. mlx4 beat
   the 8-bit reference on 3 of 6 questions. No gap worth caring about for agent use.

## Other verdicts (all measured, all in STAGES.md)

| Question | Verdict |
|---|---|
| Runtime upgrade 0.6.15 → 0.6.17 | **Adopted** — 0.6.16 fixed the hybrid-model Metal-handle leak (~47/token); patch ports cleanly |
| Drafter: MTP vs DFlash2 vs DSpark | **MTP** — DFlash2 silently bypasses APC (0% warm, 2–5× slower turns); DSpark can't load (upstream validator bug) |
| Latest-only cache: entries=1 vs 2 | **2** — entries=1 gives a 0% hit rate; 2 = latest + one live slot, and is the memory dial that governs the OOM |
| kv-bits 8 | Still rejected — streaming finish_reason bug open upstream in 0.6.17 |
| llama.cpp | Still zero hybrid cache reuse upstream (verified late-July master) — MLX only |
| Context cap | 65536 estimated tokens (40960/49152 block session start; 65536 keeps real ctx under the ~40–50k OOM zone) |

## Incidents
- 22:04 reboot: battery ran to zero (confirmed via pmset log; NOT a panic). Machine
  now on AC; keep it plugged in.
- 22:47 reboot: real kernel panic during the mlx6+APC arm (above). Panic baseline
  is now 1 file (`panic-full-2026-08-28-224750`); nothing after it.

## Worth reporting upstream (mlx-vlm)
1. DSpark drafter unloadable: validator asserts hidden_size == heads×head_dim, but
   Qwen3.8 drafter configs decouple head_dim (dspark/config.py:115).
2. DFlash2 drafter silently disables exact-mode APC (lookups never called).
3. kv-bits + streaming tool calls: stream ends without a finish_reason chunk (0.6.15,
   still unfixed in 0.6.17).
