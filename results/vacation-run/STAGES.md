# Vacation run — best-quant campaign, started 2026-08-28

Goal: best quant (8 > 6 > 4 preference) for Qwen 3.8 27B under Claude Code, warm turns
target ~4–5 s, MTP working, latest-snapshot-only caching. ALL testing through real
Claude Code sessions (`qwen-code -p` / `-c -p`) per user mandate — no synthetic replays.

Baseline: 0 panic-full files in /Library/Logs/DiagnosticReports (old 4 retired by
macOS; uptime 12d at start, 18:45). Rails: lib.sh (mem rail 25%, panic check).

## Phase 0 (18:45) — snapshot
- local-setup 2e4cfce (overnight config committed), local_llm_bench 4bff894.

## Phase 1 (18:50–19:05) — runtime upgrade PASSED
- New venv `.mlxenv-0617`: mlx 0.32.2, mlx-vlm 0.6.17 (+jinja2, mlx-lm 0.31.3, datasets).
- apply-latest-only-patch now honours LLM_MLX_VENV; applied cleanly to 0.6.17
  (anchors intact — upstream confirmed no APC-path changes 0.6.15→0.6.17).
- llm-serve MLX_PY default → .mlxenv-0617.
- Smoke (real Claude Code, mlx4, APC=2, MTP block=3): turn1 cold 126 s, turn2 261 s
  (tool-heavy), turn3 short Q **19 s**; every request warm (cached_tokens ≈ prompt,
  prefill 2.3–7.6 s @ 33–39k tok); "speculative decoding enabled"; no truncation.
- Research verdicts baked in: kv-bits OFF (streaming finish_reason bug #1850 still open
  in 0.6.17); 0.6.16 fixed ArraysCache Metal-handle leak (~47 handles/token on this
  arch — the old ~10k-token driver-crash ceiling); DSpark (RadixArk/Qwen3.8-27B-DSpark)
  and DFlash2 (z-lab/Qwen3.8-27B-DFlash2) drafters need separate weights, ~1.75×
  reported on M5 Max; llama.cpp hybrid warm cache still broken upstream — MLX only.

## Phase 2 (18:50–) — downloads in flight
- qwen3.8-27b-mlx-8bit (~29.5 GB), -6bit (~22.8 GB), dspark, dflash2 drafters.

## Phase 3 — bench ladder (planned arms, real Claude Code 12-turn sessions each)
1. mlx4 / APC=2 / MTP block=3  (0.6.17 baseline vs Aug 22)
2. mlx4 / APC=2 / DSpark
3. mlx4 / APC=2 / DFlash2
4. mlx4 / APC=1 / best drafter  (latest-only strictness test)
5. mlx6 / best config
6. mlx8 / best config
Metrics: per-turn wall, warm:cold, prefill rates, decode t/s, RSS, panic check per arm.

## 2026-08-28 19:02–19:34 — ARM 1 `v17-mlx4-mtp-e2` (PARTIAL — GPU OOM at ~75k ctx)
- config: mlx4, MTP drafter (qwen3.8-27b-mtp-mlx-8bit, kind=mtp, block=3), APC 2 entries,
  mlx-vlm 0.6.17. Drafter confirmed: "Drafter ready; speculative decoding enabled."
- rails pre: panics 0, uptime 12d 4:10, free 91%. rails post: panics 0, uptime 12d 4:42
  (continuous, NO panic), free 26%. Server survived and stayed healthy throughout.
- Results (turns 1–9 valid, turns 10–12 invalidated by OOM):
  - median turn 144 s, min 30 s, max 395 s (median over ALL 12 rows reads 108 s but that
    is an artifact of three 4–5 s failed turns — do not use it).
  - warm prefills 33/34 = 97%. Only turn 1 req 1 was cold. Cache held perfectly while it ran.
  - decode: warm mean 16.56 tok/s, cold 25.20 tok/s (n=1).
  - decode degrades monotonically with context: 23.0 t/s @31k -> 15.6 @50k -> 12.8 @75k.
  - server RSS flat 15.75–15.80 GB (no leak). free RAM min 4.20 GB (harness vm_stat metric).
- **FAILURE MODE — Metal GPU OOM, not a kernel panic and not a client bug.**
  60 requests failed with
  `[METAL] Command buffer execution failed: Insufficient Memory
   (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory)`
  raised from `apc.py:321 _clone_prompt_cache_for_apc -> mx.eval`, i.e. while APC
  deep-copies the KV snapshot in `store_exact_cache`.
  Onset 19:13:48 (~turn 3/4, ~40–50k ctx), sporadic and retried successfully through
  turn 9, then total from 19:33:28 — every request failed in ~2 s, so turns 10–12
  returned empty output in 4–5 s with ZERO completed server requests.
  Turns 8 and 9 reached the server and generated tokens (173 / 1757) but Claude Code
  printed nothing, so their .txt are empty too.
- Note: the two stderr lines in every .turn*.err (connectors-disabled warning and
  `[claude-code:unrecognized_model]`) are benign — they appear on successful turns too.
- Context: `iogpu.wired_limit_mb=51200` (50 GB of 64). The ceiling is GPU/unified-memory
  working set during snapshot cloning, NOT system RAM — the 25%-free memory_pressure rail
  never tripped and cannot detect this. Raising the wired limit needs sudo and is the
  same class of change implicated in this machine's four prior panics; NOT attempted
  autonomously. Recommend the user decide.
- Verdict: MTP+APC=2 is stable and keeps the cache 97% warm, but on mlx4 this
  12-turn accumulating session hits a GPU-memory ceiling at ~75k context. The arm is
  usable for turns 1–9. Cross-arm comparison must be restricted to comparable context.

## 2026-08-28 19:35–19:40 — ARM 2 `v17-mlx4-dspark-e2` SKIPPED (drafter will not load)
- Tried `LLM_DRAFT=$HOME/Models/qwen3.8-27b-dspark LLM_DRAFT_KIND=auto llm-serve start mlx4`
  and then the documented fallback `LLM_DRAFT_KIND=dflash`. BOTH fail identically at
  server startup ("Application startup failed. Exiting."):
```
File ".../mlx_vlm/speculative/drafters/dspark/config.py", line 115, in validate
    raise ValueError(
ValueError: DSpark hidden_size must equal attention heads * head_dim.
```
  (chain: generation.py:1201 _initialize_model -> drafters/__init__.py:174 load_drafter
   -> utils.py:819 load_model -> dspark/config.py:218 from_dict -> :68 __post_init__ -> :115 validate)
- Cause: the drafter's config.json has hidden_size=5120 but num_attention_heads=32 and
  head_dim=128, so heads*head_dim = 4096 != 5120. Qwen3 decouples head_dim from
  hidden_size, but mlx-vlm 0.6.17's DSpark validator asserts they are equal, so this
  drafter can never load under 0.6.17 as shipped.
- `LLM_DRAFT_KIND=dflash` does NOT dodge it: the weights dir carries a `dspark_config`
  key, so the loader routes to the DSpark config class regardless of the kind hint.
- Not worked around: patching mlx-vlm's validator would change the runtime under test
  mid-ladder and invalidate cross-arm comparison. Left for the user to decide (the fix is
  a one-line relaxation of that validate(), upstream).
- rails: panics 0, uptime continuous, no server left running.
- Verdict: SKIPPED — blocked by an upstream 0.6.17 validator bug, not by this machine.

## 2026-08-28 19:36–19:53 — ARM 3 `v17-mlx4-dflash2-e2` ABORTED AFTER 2 TURNS (drafter disables APC)
- config: mlx4, DFlash2 drafter, APC 2 entries. Drafter DID load:
  "Auto-detected --draft-kind='dflash' ... Drafter ready; speculative decoding enabled."
  APC also came up clean: "APC enabled (block_size=16, num_blocks=2048)" and
  "APC self-check ok mode=exact schema=v1 layers=64".
- Server RSS 18.9 GB vs 15.8 GB for MTP (+3.1 GB — the dflash2 drafter is 3.6 GB on disk),
  i.e. materially less GPU headroom before the Metal OOM ceiling found in Arm 1.
- **DFlash2 silently turns the prompt cache OFF.** Every prefill was fully cold —
  `cached_tokens=0` on 4/4 prefills (18773, 18982, 31228, 32121 tokens), never partial.
  Live /v1/cache/stats after 2 turns:
    exact_stores=0  exact_hits=0  lookups_hit=0  lookups_miss=0  matched_tokens=0
  Contrast Arm 1 (MTP) at end of run:
    exact_stores=68 exact_hits=47 lookups_hit=47 token_hit_rate=1.0 matched_tokens=2,445,723
  lookups_miss is 0 as well as lookups_hit, so APC's exact path is never even CALLED
  under the dflash drafter — it is bypassed, not missing.
- Consequence: every prefill pays full cold cost (39.9 s, 41.6 s, 85.4 s, 154.0 s, and
  166.4 s on the in-flight turn 3; prefill rate decaying 470 -> 202 tok/s as ctx grows).
  Turn walls 273 s and 504 s vs 115 s and 100 s for MTP on the identical questions.
  Warm decode 8.8 tok/s vs 20.9 tok/s for MTP on turn 2.
- ABORT rationale (deliberate deviation from "run all 12 turns"): the arm's purpose is to
  decide the best drafter, and that is already settled — MTP holds 97% warm, DFlash2 holds
  0% by construction. The remaining 10 turns would be all-cold prefills on a
  monotonically growing context, costing 60–90 min and near-certainly ending in the same
  Metal OOM, while adding nothing to the verdict. Turns 1–2 are recorded in the jsonl.
- rails: panics 0, uptime continuous, mem never below rail.
- Verdict: DFlash2 REJECTED — incompatible with APC on this build. MTP is the best drafter.

## 2026-08-28 ~19:57–22:04 — UNPLANNED REBOOT: BATTERY DEPLETION (not a panic) — RE-BASELINE
- The run was interrupted mid-Arm-4. Machine rebooted; `kern.boottime` = Fri Aug 28
  22:04:26 2026, uptime reset from "12d 5:05" to "3 mins".
- Cause CONFIRMED as battery exhaustion, corroborated independently of the user report:
  - `pmset -g log`: BatteryHealth "Warning level: 2 ... cap: 10" at 19:31:04, then
    "Warning level: 3 time: 2 cap: 2" at 19:38:56 — i.e. the battery was already at 2%
    and ~2 minutes of runtime DURING arms 1/3, and ran flat shortly after.
  - `pmset -g batt` now: "AC Power ... 2%; charging; 2:24 remaining".
  - **Zero** panic artifacts: no panic-full-*.panic, no *.panic, no *.ips anywhere in
    /Library/Logs/DiagnosticReports (nor in ~/Library/...). The four historical panics on
    this machine DID leave files, so absence here is meaningful.
  - No "Previous shutdown cause" panic record in the last 30m of the unified log.
- => NOT a kernel panic. The MTP+APC=2 config did not crash the machine; it ran clean for
  ~50 min across arms 1 and 3.
- RE-BASELINE from here: panic-file baseline remains **0**; uptime baseline is boot
  22:04:26. From this point only a NEW panic file, or a further UNEXPLAINED uptime reset,
  counts as a panic stop-condition.
- Caveat carried forward: battery is at 2% and charging. The remaining arms run on AC; if
  AC is interrupted the machine will drop again immediately. Noted, not mitigable from here.
- Arm 4 produced no data before the cut (jsonl empty) and is rerun from scratch below.
- Expect the first `llm-serve start` after reboot to be slower — model loads cold from disk.

## 2026-08-28 22:07–22:28 — ARM 4 `v17-mlx4-mtp-e1` ABORTED AFTER 3 TURNS (entries=1 breaks warmth)
- config: mlx4, MTP drafter (the Arm 1/3 winner), `LLM_APC_ENTRIES=1`. Startup clean:
  "APC enabled (block_size=16, num_blocks=2048)", "Drafter ready; speculative decoding enabled."
  RSS 15.75 GB, same as Arm 1 — the entry count does not change weight residency.
- **Warmth: entries=1 destroys it completely.** Every prefill cold, `cached_tokens=0`:
  turn 1 (18773, 0) (31019, 0); turn 2 (31777, 0); turn 3 all cold (37216, 38329, 39082 ...).
  Live /v1/cache/stats at abort: exact_stores=20, **exact_hits=0**, token_hit_rate=0.0,
  evictions=0, rejects=0 (rejects_by_reason empty).
  Contrast Arm 1 (entries=2): exact_stores=68, exact_hits=47, token_hit_rate=1.0.
- Reading: stores SUCCEED (20 of them) and nothing is rejected or evicted, but no lookup
  ever matches. A single retained slot is not enough for this access pattern — each prefill
  overwrites the one slot, and the next request's prefix no longer matches what is held, so
  the entry is always the wrong one. This is the pre-specified failure mode:
  **"entries=1 breaks warmth, keep 2".**
- Cost of losing the cache: every prefill pays full freight — 89.5 s / 92.9 s / 94.6 s at
  37–39k ctx (rate ~413 tok/s). Turn 1 264 s and turn 2 185 s, vs 115 s and 100 s for
  entries=2 on the identical questions. Decode itself is healthy (18.9–19.7 tok/s), so the
  entire regression is prefill.
- **OOM behaviour: zero Metal OOMs at entries=1** (vs 60 in Arm 1). But this is not a win —
  the OOM disappears precisely BECAUSE no snapshots are retained. It confirms the Arm 1
  diagnosis (the OOM is driven by resident APC snapshot count/size during the clone in
  `apc.py:321`), and it means entries is a real memory dial; it just cannot be turned to 1
  without turning the cache off entirely.
- ABORT rationale: warmth question conclusively answered across 3 turns / ~10 requests with
  a 0% hit rate and a mechanistic explanation; remaining 9 turns would be all-cold prefills
  costing 60–90 min and delaying the mlx6/mlx8 arms that actually answer the user's
  question. Turns 1–2 recorded in the jsonl (turn 3 cut mid-flight).
- rails: panics 0, uptime continuous since 22:04 boot, free RAM 45%, battery 6% on AC.
- Verdict: **entries=1 REJECTED. entries=2 is the config for arms 5–6.**

## 2026-08-28 22:28–22:47 — ARM 5 `v17-mlx6-mtp-e2` KERNEL PANIC — mlx6/mlx8 warm-cache DISQUALIFIED
- config: mlx6 + MTP + APC entries=2. Server started clean 22:28:39, drafter loaded,
  bench reached ~turn 3-5; machine kernel-panicked at 22:47:50.
- panic-full-2026-08-28-224750.0002.panic: `"completeMemory() prepare count underflow"
  @IOGPUMemory.cpp:550` — the SAME driver assertion as the Aug 15/16 panics.
- Mechanism now coherent across all evidence: APC's exact-store snapshot clone
  (apc.py:321 mx.eval, multi-GB per store) near the GPU wired limit. mlx4 (16.1 GB
  weights) has enough slack to fail gracefully (arm 1: request-level Metal OOMs from
  ~40-50k ctx); mlx6 (~19-23 GB) does not — the same pressure reaches the IOGPU
  refcount-underflow assert. mlx8 (+~7 GB) would be strictly worse. 0.6.17's ArraysCache
  leak fix helped the per-token path but does not govern snapshot-clone spikes.
- User informed (was at keyboard); decision: **continue mlx4-only.** No mlx6/mlx8 with
  APC ever again on this 64 GB machine. Quality grading may still run low-ctx with
  APC OFF (no snapshot churn — the panic trigger is absent; 321 prior bench rows stable).
- Mitigation shipped: qwen-code now defaults CLAUDE_CODE_MAX_CONTEXT_TOKENS=40960
  (was 262144) so sessions compact before the OOM zone. LLM_CTX still overrides.
- New panic baseline: 1 file (panic-full-2026-08-28-224750), boot 22:47:32.

## CAMPAIGN VERDICT (speed/safety portion settled 2026-08-28 22:55)
**Adopted: mlx4 + MTP block=3 + APC entries=2 + checkpoint-only patch on mlx-vlm 0.6.17.**
- Best drafter: MTP (arm 3: DFlash2 bypasses APC entirely; arm 2: DSpark cannot load).
- Entries: 2 (arm 4: entries=1 gives 0% hit rate — single slot always holds wrong entry).
- Quant: mlx4 — fastest measured AND the only quant that can hold a warm cache safely.
  8-bit/6-bit preference is overruled by the panic evidence, not by speed alone.
- Remaining: low-ctx quality grading (quant cost curve, APC off), verification session,
  docs, commits.
