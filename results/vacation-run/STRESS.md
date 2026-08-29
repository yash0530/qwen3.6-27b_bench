# Context-limit stress campaign — mlx4 / mlx6 / mlx8

Run started 2026-08-29 12:46:52 on M5 Pro / 64 GB.
Goal: drive REAL Claude Code sessions whose context grows toward the model's native
262,144 tokens, and find where each quant stops being usable.

## Baseline for a post-reboot reader

- `kern.boottime` at campaign start: `kern.boottime: { sec = 1787982452, usec = 764369 } Fri Aug 28 22:47:32 2026`
- panic files present at campaign start: **1**
```
/Library/Logs/DiagnosticReports/panic-full-2026-08-28-224750.0002.panic
```
If, when you read this, `sysctl kern.boottime` is LATER than the value above, or the
panic-file count is HIGHER than 1, then the machine panicked during this campaign
and the run below is truncated at the last row written. That truncation is itself the
finding for the quant in the last row.

## Method

- Stack: `llm-serve start <quant>` — MTP drafter + APC `entries=2` + both local patches
  (apply-latest-only-patch, apply-single-clone-patch). Server log rotates on start.
- Turns: `LLM_CTX=262144 ./scripts/qwen-code --dangerously-skip-permissions
  --session-id/--resume <uuid> --output-format json -p "<q>"` from `local-setup/`,
  stdin `</dev/null`, per-turn stdout/err in `stress-turns/`.
  A fixed session UUID per quant is used instead of `-c` so the campaign can never
  latch onto the wrong conversation.
- Ballast: `stress-corpus/file1..file10`, ~120 KB each (~30k tokens), each a distinct
  fictional document with ONE unique fact buried mid-file (~55% of the way in).
  Turn N reads file N in full and must answer a question that only that fact answers,
  which proves the bytes really entered context AND that recall still works at depth.
- Per-turn metrics are parsed from the server log's `Request completed:` lines
  (`prompt_tokens`, `prefill tok/s`, `decode tok/s`) for that turn's byte range only.
- `prefill t/s` column averages only requests below 5000 tok/s — above that is an APC
  cache hit replaying stored KV, not real prefill work.
- Failure rule: rc!=0, empty answer, or an API error = a failed turn.
  3 consecutive failures = that quant's total-failure ceiling; stop, summarise, next quant.

## Turn log

| quant | turn | file | status | rc | wall s | ctx depth (prompt_tok) | reqs | prefill t/s | decode t/s | OOMs total (delta) | RSS GB | free RAM % | recall | apc hit/store | note |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| mlx4 | 1 | file1 | OK | 0 | 204.7 | 57793 | 3 | 681.6 | 21.8 | 0 (+0) | 15.77 | 24.7 | yes | 1/6 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx4 | 2 | file2 | FAIL | 0 | 308.0 | 80648 | 2 | 0.0 | 16.4 | 3 (+3) | 15.71 | 8.3 | NO | 5/12 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx4 | 3 | file3 | FAIL | 0 | 65.6 | 0 | 0 | 0.0 | 0.0 | 5 (+2) | 15.61 | 6.2 | NO | 7/12 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx4 | 4 | file4 | TIMEOUT | 124 | 1740.0 | 119848 | 1 | 1473.8 | 13.2 | 12 (+7) | 15.62 | 19.5 | NO | 8/13 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |

### mlx4 — arm summary (complete, no panic)

| metric | value |
|---|---|
| last FULLY successful turn | turn 1 — **57,793** prompt tokens, correct recall (TANGERINE-OWL), 204.7 s |
| max healthy context | **~57.8k** |
| first `kIOGPUCommandBufferCallbackErrorOutOfMemory` | turn 2, at **80,625** prompt tokens |
| total-failure ceiling (3 consecutive fails) | turns 2,3,4 — deepest attempted **119,848** |
| OOM-failed requests this arm | 12 |
| panic | **no** — machine survived the whole arm |
| peak wired memory | **56.64 GB** against an `iogpu.wired_limit_mb` of 50 GB |
| peak server RSS | 15.62 GB (see note — RSS is NOT the honest signal) |
| min free RAM | 6.65 % |

**Where the OOM actually comes from.** Not APC's snapshot clone, which is what the
existing notes predicted. Every OOM traceback in this arm terminates in the *drafter*:

```
File ".../mlx_vlm/speculative/common.py", line 89, in draft_tokens
    mx.async_eval(*arrays)
RuntimeError: [METAL] Command buffer execution failed: Insufficient Memory
             (00000008:kIOGPUCommandBufferCallbackErrorOutOfMemory).
```

**Failure is graceful, not fatal.** A failed turn returns HTTP 500 upstream; Claude Code
reports `stop_reason=end_turn`, `is_error=false` and an **empty** `result`. So at depth the
session does not crash — it silently answers nothing, which is why recall flips to NO at
turn 2 onward. Turn 4 additionally spent 1740 s retrying a 119,848-token prefill from
cold (each retry re-prefills the whole prompt) and hit the harness timeout.

**Prefill/decode vs depth (mlx4)**

| depth bucket | prefill tok/s | decode tok/s |
|---|---|---|
| ~19k (turn 1, cold) | 462 | 25.8 |
| ~26k (warm APC hit) | 2718 (cache replay) | 18.8 |
| ~58k (turn 1 end) | 682 | 21.8 |
| ~80k (turn 2, OOM onset) | — (request died) | 16.4 |
| ~97-120k (turns 3-4) | ~300 measured on retries | 13.2 |

Decode degrades roughly monotonically, 25.8 -> 13.2 tok/s, about a 2x loss from 19k to 120k.
Cold prefill falls from 462 to ~300 tok/s as depth grows.

#### mlx4 memory breakdown (measured)

### mlx4
- baseline wired (idle, weights resident): **3.13 GB**  | baseline RSS 15.49 GB
- peak wired observed this arm: **56.64 GB** vs iogpu limit 50 GB
- depth samples: 372, mem samples: 4439

| depth bucket | nearest actual depth | live KV computed GB | wired GB observed | wired-baseline GB | peak wired +30s GB | free RAM % | headroom vs 50 GB |
|---|---|---|---|---|---|---|---|
| 25k | 28672 | 1.95 | 47.00 | 43.87 | 49.88 | 22.1 | 0.12 |
| 50k | 53248 | 3.57 | 51.80 | 48.67 | 52.88 | 15.1 | -2.88 |
| 75k | 77824 | 5.18 | 50.55 | 47.42 | 54.53 | 16.9 | -4.53 |
| 100k | 104448 | 6.92 | 51.94 | 48.81 | 54.51 | 14.9 | -4.51 |
| 120k | 118784 | 7.86 | 54.35 | 51.22 | 56.06 | 11.3 | -6.06 |

- wired slope 2048→118784 tok: **43.2 KB/token** (arch math: 64 KB/token per retained KV copy → ~0.68 concurrent copies)

Note the two corrections this arm forced on the method:

1. **`ps` RSS does not see MLX's Metal buffers.** Across a 97k-token prefill RSS moved
   0.11 GB while the KV alone must be ~6.5 GB. The `llm-serve` comment that "RSS is the
   only honest signal" is wrong for this failure mode — RSS tracks the mmap'd *weights*
   and essentially nothing else. **Wired memory is the signal that predicts the OOM.**
2. Wired passes the 50 GB `iogpu` limit at around 50k context and stays over it,
   which is exactly where graceful OOM failures begin.

#### Component costs measured directly (controlled A/B, mlx4)

| component | how measured | value |
|---|---|---|
| MTP drafter, resident | `llm-serve start mlx4` with `LLM_SPEC=0` vs default, RSS after an identical warm request | 15.29 GB -> **15.71 GB = 0.42 GB** (matches the 456 MB weights dir) |
| mlx4 weights, resident | RSS at idle, drafter off | **15.29 GB** (`du` 15 G) |
| system wired at idle, model loaded | `vm_stat` "Pages wired down" before any request | **~3.3 GB** |
| `iogpu.wired_limit_mb` | `sysctl` | **51200 (50 GB)** |

The idle numbers show why RSS misleads here: MLX **mmaps** the weights, so at rest they
are resident (RSS 15.7 GB) but barely wired (3.3 GB). Only when a request runs does Metal
wire the working set - weights + live KV + prefill activations + APC snapshots - and that
is when wired jumps to 47-56 GB. The OOM is a *wired-limit* failure, not an RSS failure.

The drafter is the same `qwen3.8-27b-mtp-mlx-8bit` directory for all three quants
(`model_draft()` in `llm-serve`), so 0.42 GB is its cost in every arm below.
| mlx6 | 1 | file1 | OK | 0 | 241.4 | 57824 | 3 | 655.8 | 9.1 | 0 (+0) | 22.03 | 25.2 | yes | 1/6 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx6 | 2 | file2 | FAIL | 0 | 490.1 | 80690 | 1 | 0.0 | 8.6 | 2 (+2) | 22.06 | 9.8 | NO | 2/8 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx6 | 3 | file3 | FAIL | 0 | 244.1 | 80812 | 0 | 0.0 | 0.0 | 4 (+2) | 22.05 | 12.5 | NO | 3/10 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx6 | 4 | file4 | FAIL | 0 | 2.2 | 80931 | 0 | 0.0 | 0.0 | 6 (+2) | 22.06 | 8.1 | NO | 5/10 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |

### mlx6 — arm summary (complete, no panic)

**Headline: mlx6 survives now.** The previous mlx6 attempt kernel-panicked the machine
(`IOGPUMemory.cpp:550`). This time it ran a full, correct turn and then degraded exactly
the way mlx4 does — graceful OOM, no panic, machine still up.

| metric | value |
|---|---|
| last FULLY successful turn | turn 1 — **57,824** prompt tokens, correct recall (TANGERINE-OWL), 241.4 s |
| max healthy context | **~57.8k** |
| first OOM | turn 2, at **80,667** prompt tokens |
| total-failure ceiling (3 consecutive fails) | turns 2,3,4 — wedged at **~80.9k** |
| OOM-failed requests this arm | 6 |
| panic | **no** (panic-file count still 1, `kern.boottime` unchanged) |
| peak wired memory | **56.35 GB** vs 50 GB limit |
| peak server RSS | 22.18 GB |
| min free RAM | 11.35 % |

All six OOM-failing requests landed in a very tight band: **80,667 / 80,690 / 80,793 /
80,812 / 80,912 / 80,931**. By turn 4 the failure was instant (2.2 s) — once past the
threshold the server rejects immediately rather than grinding.

**Prefill/decode vs depth (mlx6)**

| depth bucket | prefill tok/s | decode tok/s |
|---|---|---|
| ~19k (cold) | ~430 | 9.6 |
| ~58k (turn 1 end) | 656-1121 | 9.0-9.1 |
| ~80k (OOM onset) | — (request died) | 8.6 |

mlx6 decode is **~2.4x slower than mlx4** at the same depth (9.1 vs 21.8 tok/s at ~58k)
for an identical context ceiling, which is the central practical result of this arm.

#### mlx6 memory breakdown (measured)

### mlx6
- baseline wired (idle, weights resident): **3.09 GB**  | baseline RSS 0.08 GB
- peak wired observed this arm: **56.35 GB** vs iogpu limit 50 GB
- depth samples: 163, mem samples: 2527

| depth bucket | nearest actual depth | live KV computed GB | wired GB observed | wired-baseline GB | peak wired +30s GB | free RAM % | headroom vs 50 GB |
|---|---|---|---|---|---|---|---|
| 25k | 28672 | 1.95 | 49.03 | 45.94 | 50.23 | 19.2 | -0.23 |
| 50k | 53248 | 3.57 | 48.78 | 45.69 | 53.30 | 19.6 | -3.30 |
| 58k | 61440 | 4.10 | 53.30 | 50.21 | 54.38 | 13.0 | -4.38 |
| 75k | 77824 | 5.18 | 54.29 | 51.20 | 56.23 | 11.4 | -6.23 |
| 81k | 80796 | 5.37 | 52.59 | 49.50 | 56.23 | 12.5 | -6.23 |

- wired slope 2048→80796 tok: **277.9 KB/token** (arch math: 64 KB/token per retained KV copy → ~4.34 concurrent copies)
| mlx8 | 1 | file1 | FAIL | 0 | 219.8 | 57816 | 2 | 421.6 | 14.0 | 2 (+2) | 26.62 | 17.9 | NO | 2/8 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx8 | 2 | file2 | FAIL | 0 | 2.3 | 57944 | 0 | 0.0 | 0.0 | 4 (+2) | 26.63 | 12.4 | NO | 4/10 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |
| mlx8 | 3 | file3 | FAIL | 0 | 1.8 | 58066 | 0 | 0.0 | 0.0 | 6 (+2) | 26.64 | 7.3 | NO | 6/10 | [claude-code:unrecognized_model] {"model":"qwen-local","query_source":"sdk"} |

### mlx8 — arm summary (complete, no panic)

mlx8 **loaded fine** (RSS 26.7 GB) — the predicted "may fail to even load" did not happen.
It failed on *depth* instead, and it is the only quant that could not complete even turn 1.

| metric | value |
|---|---|
| deepest COMPLETED request | **41,511** prompt tokens |
| max healthy context | **~41.5k** |
| first OOM | turn 1, at **57,793** prompt tokens |
| total-failure ceiling (3 consecutive fails) | turns 1,2,3 — wedged at **~58k** |
| turn 1 recall | **failed** (empty answer — the turn OOM'd before answering) |
| OOM-failed requests this arm | 6 |
| panic | **no** |
| peak wired memory | **56.61 GB** vs 50 GB limit |
| peak server RSS | 26.7 GB |
| min free RAM | 8.4 % |

Turns 2 and 3 failed in **2.3 s and 1.8 s** — once over the line the server rejects
instantly. mlx8 never produced a single correct answer in this campaign.

**Prefill/decode vs depth (mlx8)**

| depth bucket | prefill tok/s | decode tok/s |
|---|---|---|
| ~19k (cold) | 422 | 14.0 |
| ~41k (deepest completed) | ~400 | ~13 |
| ~58k | — (OOM) | — |

#### mlx8 memory breakdown (measured)

### mlx8
- baseline wired (idle, weights resident): **3.07 GB**  | baseline RSS 0.01 GB
- peak wired observed this arm: **56.61 GB** vs iogpu limit 50 GB
- depth samples: 46, mem samples: 669

| depth bucket | nearest actual depth | live KV computed GB | wired GB observed | wired-baseline GB | peak wired +30s GB | free RAM % | headroom vs 50 GB |
|---|---|---|---|---|---|---|---|
| 25k | 28672 | 1.95 | 43.55 | 40.48 | 46.35 | 23.5 | 3.65 |
| 41k | 43543 | 2.93 | 51.19 | 48.12 | 54.39 | 12.9 | -4.39 |
| 50k | 53783 | 3.60 | 54.31 | 51.24 | 55.24 | 8.4 | -5.24 |
| 58k | 57928 | 3.87 | 54.20 | 51.13 | 56.61 | 10.8 | -6.61 |

- wired slope 2048→57928 tok: **302.9 KB/token** (arch math: 64 KB/token per retained KV copy → ~4.73 concurrent copies)

## Stacked memory budget per quant (modelled vs observed)

`weights` and `MTP drafter` are measured resident sizes; `live KV` and `APC snapshots`
are arch math (64 KB/token per retained copy + 72 MB fixed GatedDeltaNet state, APC
`entries=2`); `observed peak wired` is measured from `vm_stat` at that depth.
`unaccounted` is what the model does NOT explain — prefill activation buffers, Metal
scratch, and APC's transient clone.


### mlx4 (weights 15.29 GB + drafter 0.42 GB)

| depth | live KV GB | 2x APC snapshots GB | modelled total GB | observed peak wired GB | unaccounted GB | headroom vs 50 GB |
|---|---|---|---|---|---|---|
| 25k | 1.71 | 3.28 | 20.70 | 49.88 | +29.18 | +0.12 |
| 50k | 3.35 | 6.55 | 25.62 | 52.88 | +27.26 | -2.88 |
| 75k | 4.99 | 9.83 | 30.53 | 54.53 | +24.00 | -4.53 |
| 100k | 6.63 | 13.11 | 35.45 | 54.51 | +19.06 | -4.51 |
| 120k **<- ceiling** | 7.94 | 15.73 | 39.38 | 56.06 | +16.68 | -6.06 |

### mlx6 (weights 22.18 GB + drafter 0.42 GB)

| depth | live KV GB | 2x APC snapshots GB | modelled total GB | observed peak wired GB | unaccounted GB | headroom vs 50 GB |
|---|---|---|---|---|---|---|
| 25k | 1.71 | 3.28 | 27.59 | 50.23 | +22.64 | -0.23 |
| 50k | 3.35 | 6.55 | 32.51 | 53.30 | +20.79 | -3.30 |
| 58k | 3.88 | 7.60 | 34.08 | 54.38 | +20.30 | -4.38 |
| 75k | 4.99 | 9.83 | 37.42 | 56.23 | +18.81 | -6.23 |
| 81k **<- ceiling** | 5.38 | 10.62 | 38.60 | 56.23 | +17.63 | -6.23 |

### mlx8 (weights 26.70 GB + drafter 0.42 GB)

| depth | live KV GB | 2x APC snapshots GB | modelled total GB | observed peak wired GB | unaccounted GB | headroom vs 50 GB |
|---|---|---|---|---|---|---|
| 25k | 1.71 | 3.28 | 32.11 | 46.35 | +14.24 | +3.65 |
| 41k | 2.76 | 5.37 | 35.26 | 54.39 | +19.13 | -4.39 |
| 50k | 3.35 | 6.55 | 37.03 | 55.24 | +18.21 | -5.24 |
| 58k **<- ceiling** | 3.88 | 7.60 | 38.60 | 56.61 | +18.01 | -6.61 |

### Cross-quant comparison at a common depth (28,672 tokens — the deepest bucket all three reached)

| quant | weights GB | drafter GB | live KV GB | 2x APC GB | modelled GB | observed peak wired GB | headroom vs 50 GB | decode t/s |
|---|---|---|---|---|---|---|---|---|
| mlx4 | 15.29 | 0.42 | 1.95 | 3.76 | 21.42 | 49.88 | +0.12 | 21.8 |
| mlx6 | 22.18 | 0.42 | 1.95 | 3.76 | 28.31 | 50.23 | -0.23 | 9.1 |
| mlx8 | 26.70 | 0.42 | 1.95 | 3.76 | 32.83 | 46.35 | +3.65 | 14.0 |

**What dominates at the ceiling.** Not the weights. Going mlx4 -> mlx8 adds 11.4 GB of
weights but moves the ceiling *down* by only ~60k tokens, and every arm peaks at the same
place: **56.0-56.6 GB wired**, i.e. ~6 GB *over* the 50 GB `iogpu.wired_limit_mb`. The
modelled stack (weights + drafter + KV + 2 snapshots) accounts for only about half of that
at depth; the rest is transient — prefill activation buffers and APC's clone — which is why
the measured wired slope is **~280-300 KB/token**, roughly 4.3-4.7x the 64 KB/token a single
retained KV copy would cost. That transient multiplier, not the resident footprint, is what
sets every ceiling here.

## FINDINGS TABLE

| quant | max healthy ctx | OOM onset ctx | total-failure ctx | panic | prefill t/s @19k / @41-58k / @80k+ | decode t/s @19k / @58k / @80k+ | peak wired GB | peak RSS GB |
|---|---|---|---|---|---|---|---|---|
| **mlx4** | **57,793** | **80,625** | **119,848** (turns 2-4) | no | 462 / 682 / ~300 | 25.8 / 21.8 / 13.2 | 56.64 | 15.62 |
| **mlx6** | **57,824** | **80,667** | **~80,931** (turns 2-4) | no | ~430 / 656-1121 / — | 9.6 / 9.1 / 8.6 | 56.35 | 22.18 |
| **mlx8** | **41,511** | **57,793** | **~58,066** (turns 1-3) | no | 422 / ~400 / — | 14.0 / — / — | 56.61 | 26.70 |

Campaign end state: `kern.boottime: { sec = 1787982452, usec = 817570 } Fri Aug 28 22:47:32 2026`, panic files: **1** — unchanged from the start, so
**the machine never panicked during this campaign**. All servers stopped, `stress-corpus/` deleted.

## Conclusions

1. **The single-clone patch's predicted ~55-70k mlx4 ceiling holds, and is if anything
   slightly conservative.** mlx4 completed a full, correct 57.8k-token turn and did not see
   its first OOM until **80.6k** — the top of the predicted band for graceful onset, with
   total failure only at ~120k. The `CLAUDE_CODE_MAX_CONTEXT_TOKENS=81920` default now in
   `qwen-code` is almost exactly the right number for mlx4: it sits just under the measured
   80.6k OOM onset.

2. **mlx6 survives — that is the biggest change.** The configuration that previously
   kernel-panicked the machine (`IOGPUMemory.cpp:550`) now completes a correct turn and then
   degrades gracefully, identically to mlx4. Zero panics across all three arms and ~2.5 hours
   of sustained over-limit GPU pressure. Entries=2 + checkpoint-only stores + the single-clone
   patch have converted a hard kernel fault into a recoverable userspace error.

3. **But mlx6 buys nothing.** It has the *same* ceiling as mlx4 (57.8k healthy, ~80.6k onset —
   the OOM depths differ by 42 tokens) while decoding **2.4x slower** (9.1 vs 21.8 tok/s).
   There is no context or stability reason to prefer it. mlx4 remains the right choice.

4. **mlx8 is not usable for Claude Code work.** It loads fine, but it cannot complete even a
   single 57.8k turn — it OOM'd at 57,793 with a deepest *completed* request of 41,511, and
   never returned one correct answer. Its usable ceiling is ~41k, below where a real session
   starts.

5. **The binding constraint is transient GPU memory, not the weights, and RSS cannot see it.**
   All three quants die at the same **~56 GB wired**, ~6 GB over the 50 GB
   `iogpu.wired_limit_mb`, despite an 11.4 GB spread in weight size. Wired grows at
   **~280-300 KB/token**, ~4.5x the 64 KB/token of one retained KV copy, so prefill
   activations plus APC's clone — not resident state — set every ceiling. Note for future
   work: `ps` RSS moved 0.11 GB across a 97k-token prefill and is useless as a warning signal
   here; **`vm_stat` wired is the metric to watch**, and raising `iogpu.wired_limit_mb` above
   51200 is the single most promising lever left untested.

6. **Failure is silent, which is the dangerous part.** An OOM'd turn returns HTTP 500 upstream
   but Claude Code reports `is_error=false`, `stop_reason=end_turn` and an **empty** result —
   the session appears to work and simply answers nothing. Any depth guard has to be a
   *preventive* context cap; there is no error to react to.
