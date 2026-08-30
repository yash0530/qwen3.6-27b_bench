# GGUF stress campaign — gguf4 / gguf5 / gguf6 (llama.cpp build 10621)

Run started 2026-08-29 14:09 PDT on M5 Pro / 64 GB.
Companion to `STRESS.md` (the MLX arms). Same machine, same model family
(Qwen3.8-27B), same real-Claude-Code method — different inference engine.

**Headline question:** llama.cpp **build 10621** (upgraded from 9620) includes the
hybrid-checkpoint fixes PR #24176 / #25472. Under build 9620 a Claude Code session on
this model measured **0/29 warm cache hits** — every turn re-prefilled from cold.
Does the warm cache work now?

## Baseline for a post-reboot reader

- `kern.boottime` at campaign start: `kern.boottime: { sec = 1787982452, usec = 817570 } Fri Aug 28 22:47:32 2026`
- panic files present at campaign start: **1**
```
/Library/Logs/DiagnosticReports/panic-full-2026-08-28-224750.0002.panic
```
If, when you read this, `sysctl kern.boottime` is LATER than the value above, or the
panic-file count is HIGHER than 1, the machine panicked during this campaign and the
run below is truncated at the last row written. That truncation is itself the finding
for the quant in the last row.

- `iogpu.wired_limit_mb` = **51200 (50 GB)** (unchanged from the MLX campaign)
- llama.cpp: `version: 0.3.0 (build 10621, commit c1d0e7a00)`, Homebrew, Darwin arm64
- wired at idle before anything started: 202,810 pages x 16 KB = **3.10 GB**

## Method

- Stack: `llm-serve start <quant>` -> `llama-server -m <gguf> --spec-type draft-mtp
  -c 262144 -ngl 99 -fa on -np 1 --cache-reuse 256 --slot-save-path ... --jinja
  --reasoning-format deepseek --temp 1.0 --top-p 0.95 --top-k 20`, plus the
  Anthropic<->OpenAI proxy on 8790. Server log rotates on start.
- Quants: `gguf4` = Qwen3.8-27B-UD-Q4_K_XL (17.6 GB file), `gguf5` = UD-Q5_K_XL
  (20.9 GB), `gguf6` = UD-Q6_K_XL (downloading at campaign start).
- **Phase A (warmth)** — 6 short repo questions through a real Claude Code session
  (`./scripts/qwen-code --dangerously-skip-permissions -p`, then `-c -p` continuations,
  stdin `</dev/null`). Per turn: wall clock, and from the server log whether the prompt
  was processed in full or restored from a context checkpoint.
- **Phase B (depth ladder)** — `stress-corpus/file1..file10`, ~120 KB each (~30k tokens),
  each a distinct synthetic document with ONE unique planted fact mid-file. Turn N reads
  file N and must answer a question only that fact answers. `LLM_CTX=262144`.
  `vm_stat` wired + server RSS sampled at each depth bucket.
- **Phase C (speed)** — prefill/decode tok/s from the server's own timing lines at
  shallow/mid/deep, plus MTP draft acceptance where logged.
- Failure rule: rc!=0, empty answer, or an API error = a failed turn.
  3 consecutive failures = that quant's ceiling; stop, summarise, next quant.

## Progress log

- 14:09 — baseline recorded. No server running. gguf6 still downloading
  (`snapshot_download` of `unsloth/Qwen3.8-27B-GGUF`, pid 62591); gguf4 and gguf5 present
  and size-stable.

---

# ARM 1 — gguf4 (Qwen3.8-27B-UD-Q4_K_XL, 17.6 GB file)

Server start 14:12. Startup is ~4 s (vs minutes for MLX): weights are mmap'd and the
KV cache for `-c 262144` is preallocated immediately.

### Startup log — two lines that matter

```
common_speculative_init_result: creating MTP draft context against the target model
srv load_model: cache_reuse is not supported by this context, it will be disabled
```

1. The inline MTP head loads (no separate draft model file) — good.
2. **`--cache-reuse 256` is silently DISABLED.** This model is hybrid/recurrent
   (GatedDeltaNet layers), and llama.cpp's cache-reuse (the "skip a gap in the middle of
   the prompt" optimisation) is not implemented for such a context. So everything below
   is achieved by plain longest-common-prefix slot matching + the build-10621 hybrid
   checkpoints, *not* by `--cache-reuse`. That flag can be dropped from `llm-serve` for
   these models; it does nothing.

Idle-after-load footprint: **RSS 35.0 GB, wired 39.61 GB** with zero requests served.
llama.cpp preallocates the whole 262,144-token KV up front, so ~36.5 GB of wired is
committed before the first token — see the memory section.

## Phase A — warmth (THE decisive question)

Real Claude Code session, fixed session id `805c54cf-…`, `-p` then `--resume … -p`,
stdin `</dev/null`, short repo questions from `local-setup/`.

| turn | question | wall s | reqs | prompt tok | **cached tok** | **cached %** | reprocessed | prefill t/s | decode t/s | draft acc | verdict |
|---|---|---|---|---|---|---|---|---|---|---|---|
| A1 | what does llm-serve do? | 135 | 2 | 18,368 / 30,568 | 0 / 18,127 | 0% / 59% | 18,368 / 12,445 | 295 / 217 | 17.3 / 13.6 | 0.957 / 0.680 | cold (first ever) |
| A2 | list every model alias | **17** | 1 | 30,936 | **30,805** | **99.6%** | 131 | 102 | 15.9 | 0.886 | **WARM** |
| A3 | which alias -> which engine | **8** | 1 | 31,185 | 30,931 | 99.2% | 254 | 139 | 16.5 | 0.939 | **WARM** |
| A4 | proxy port / server port | **8** | 1 | 31,289 | 31,181 | 99.7% | 108 | 98 | 15.0 | 0.803 | **WARM** |
| A5 | read qwen-code, explain ctx cap | 44 | 2 | 31,387 / 32,650 | 31,287 / 31,383 | 99.7% / 96.1% | 100 / 1,267 | 90 / 185 | 16.4 / 14.2 | 1.000 / 0.759 | **WARM** |
| A6 | name the two MLX patch scripts | 100 | 4 | 33.1k-34.3k | 32.6k-34.0k | 98.8% avg | 153-839 | 123-173 | 12.3-16.5 | 0.61-0.95 | **WARM** |

All six turns answered correctly (verified against the repo).

### Phase A verdict — **WARM CACHE WORKS. This is the headline finding.**

| metric | build 9620 (previous campaign) | build 10621 (this campaign) |
|---|---|---|
| warm prefix hits under Claude Code | **0 / 29** | **11 / 11** continuation requests |
| mean cached fraction on a continuation | ~0% | **0.988** |
| short follow-up turn wall clock | full re-prefill every time | **8-17 s** |

Every request after the very first one selected the existing slot by LCP similarity with
`f_sim_best` 0.96-0.997 and re-processed only 100-1,300 tokens out of a ~31-34k prompt.
The two-request A1 turn shows it working *within* a turn too: request 2 of turn 1 kept
18,127 of its 30,568 tokens.

Note there is no `restored context checkpoint` line and no `forcing full prompt
re-processing` line anywhere in the log. In build 10621 the hybrid path simply keeps the
slot's recurrent state valid and matches by prefix; the PR #24176/#25472 fixes show up as
the *absence* of the forced-reprocess line, not as a new positive log message. The
observable is `f_keep` in `selected slot by LCP similarity` — that is the number to watch.

Practical consequence: an 8-second follow-up on a 31k context. Under MLX the same
follow-up costs a full re-prefill (mlx4 measured 204 s for a 57.8k turn).

### Phase A memory behaviour — the shape is completely different from MLX

| | MLX (mlx4) | llama.cpp (gguf4) |
|---|---|---|
| wired at idle, model loaded | 3.13 GB | **39.61 GB** |
| wired under load | climbs with depth, 47 -> 56.6 GB | **flat, 39.6-39.8 GB at every depth** |
| RSS | ~flat 15.6 GB (useless signal) | **grows with depth, 35.0 -> 47.6 GB** |
| KV allocation | grown lazily per request | **preallocated in full for `-c 262144` at load** |

This inverts the MLX campaign's central lesson. Under MLX, RSS was blind and `vm_stat`
wired predicted the OOM. Under llama.cpp the opposite holds: **wired is a constant**
(the whole 262k KV is committed at load, so there is nothing left to grow) and **RSS is
the signal that tracks depth** — it is the CPU-side prompt cache / checkpoint store
(`--cache-ram`, default 8192 MiB) plus page-in of the mmap'd weights.

Wired at idle 39.61 GB - 3.10 GB system baseline = **36.5 GB committed by the server**,
against a 17.6 GB weights file: ~18.9 GB of that is the preallocated 262,144-token KV
plus the MTP draft context, i.e. **~72 KB/token** — the same per-token KV cost the MLX
arithmetic predicted. llama.cpp just pays it all up front instead of on demand.

Consequence: **gguf4 at `-c 262144` sits at 39.6 GB wired against the 50 GB
`iogpu.wired_limit_mb` before it has served a single token, and stays there.** There is
~10 GB of wired headroom, and it does not shrink with context depth. Note also that wired
drops back to ~3.6 GB whenever the server goes idle — Metal releases the residency set.

### An important negative: the prefix collapses at depth

Not every deep request stays warm. Measured on the first depth-ladder attempt:

```
task 1225  prompt= 65,201  cached= 65,140 (100%)  f_keep=0.999  ->    1 s prefill
task 1250  prompt= 91,803  cached= 18,339 ( 20%)  f_keep=0.288  ->  574 s prefill
task 1340  prompt=110,216  cached= 91,798 ( 83%)  f_keep=0.998  ->  232 s prefill
```

The collapse lands on **exactly 18,339 tokens** — the boundary just after the system
prompt and before the first tool result. Every deep-ladder turn that reads a large file
shows one such request. So the honest warmth verdict has two halves:

- **Conversational turns (no big new tool output): reliably warm** — 11/11 in Phase A,
  8-17 s each.
- **Turns that append tens of thousands of tokens of tool output: one request per turn
  falls all the way back to the system-prompt boundary and re-prefills everything after
  it.** At 92k that cost 574 s.

`--ctx-checkpoints` defaults to 32 and `--checkpoint-min-step` to 8192 (so checkpoints
should span the full 262k), and the log emits **zero** checkpoint lines at verbosity 3 —
the only cache observable llama.cpp gives here is `f_keep` on the
`selected slot by LCP similarity` line. That is the number to instrument.

### gguf6 availability

`Qwen3.8-27B-UD-Q6_K_XL.gguf` finished downloading at 14:15 — 25,299,061,664 bytes
(23.6 GiB); the `snapshot_download` process has exited and the size was identical on two
checks 60 s apart (`25299061664 Aug 29 14:15:30` both times), so the file is complete.

### KV arithmetic, read straight off the GGUF header

| key | value |
|---|---|
| `qwen35.block_count` | 65 |
| `qwen35.full_attention_interval` | 4 -> **16 full-attention layers**, 49 GatedDeltaNet layers |
| `qwen35.attention.head_count_kv` | 4 |
| `qwen35.attention.key_length` / `value_length` | 256 / 256 |
| `qwen35.ssm.state_size` / `conv_kernel` / `inner_size` | 128 / 4 / 6144 |
| `qwen35.nextn_predict_layers` | 1 (the inline MTP head) |
| `qwen35.context_length` | 262,144 |

KV per token = 4 kv-heads x (256 + 256) x 2 B x 16 attention layers = **65,536 B = 64 KB/token**,
which is exactly the figure the MLX campaign derived. At `-c 262144` that is **16.0 GiB**
of KV, preallocated. 16.0 GiB KV + 16.35 GiB weights = 32.4 GiB = 34.7 GB, against the
**36.5 GB** actually committed — the ~1.8 GB remainder is the SSM/conv states, compute
buffers and the MTP draft context. The model closes.

**This is why llama.cpp's `-c` is the memory dial and MLX's is not.** Dropping `-c` to
131072 would hand back 8 GiB of wired outright.

## Phase B — depth ladder (gguf4)

Fresh session `cd17dd57-…`, `LLM_CTX=262144`, turn N reads `stress-corpus/fileN.md`
(~120 KB, ~28k tokens) and must answer that file's planted-fact question.
Note Claude Code's Read tool truncates a file at ~75 KB and the model then issues a
second Read with an offset, so **one corpus file costs two tool results and ~46k tokens
of context**, not ~28k. The ladder therefore climbs ~46k per turn, not ~25k.

| turn | file | status | wall s | reqs | ctx depth reached | cached frac | prefill s | recall | RSS GB | wired GB | free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | file1 | OK | 312 | 3 | **65,146** | 0.637 | 294 | **yes** (TANGERINE-LYNX) | 45.15 | 40.21 | 10.80 |

### ROOT CAUSE of the deep-context collapse: `--cache-ram` is far too small (default 8 GiB)

The server says it outright. Three lines, all from the `srv alloc` subsystem:

```
W srv alloc:  - making room for prompt cache entry, removing oldest entry (size = 3518.227 MiB)
W srv alloc:  - making room for prompt cache entry, removing oldest entry (size = 6655.967 MiB)
W srv alloc:  - prompt state size 9774.471 MiB exceeds cache size limit 8192.000 MiB, skipping
```

llama.cpp keeps a **prompt cache** of saved slot states (`-cram, --cache-ram`, default
**8192 MiB**). When a request cannot continue the live slot, warmth depends on restoring
a state from that cache. For this model a saved state costs:

| slot depth | saved prompt-state size | implied cost |
|---|---|---|
| ~34k tok | 3,518 MiB | 104 KB/token |
| ~65k tok | 6,656 MiB | 103 KB/token |
| ~95k tok | 9,774 MiB | 103 KB/token |

**~103 KB/token** — the 64 KB/token of attention KV plus the GatedDeltaNet conv/SSM state
for all 65 blocks, which is the price of the hybrid architecture. Against the 8,192 MiB
default that gives a hard arithmetic ceiling:

> **8192 MiB / 103 KB per token = ~79,000 tokens.** Past ~79k, a slot state is
> *never* cacheable — the server logs `exceeds cache size limit ... skipping` and the
> state is thrown away. Below ~79k only **one** deep entry fits at a time, so any second
> conversation evicts the first.

That is exactly the collapse measured earlier: the `f_keep=0.288` request at 92k, and the
`f_keep=0.170` request that opened the fresh ladder session, both follow a
`removing oldest entry` / `skipping` line. It is not a hybrid-checkpoint bug and not a
regression — **it is a default that was never sized for a 27B hybrid at 64 KB/token.**

**Actionable:** `llm-serve` should pass `-cram` explicitly for the gguf aliases. At
`-cram 24576` (24 GiB) three ~79k states fit, or one ~240k state. The tradeoff is host
RAM, not wired GPU memory — the prompt cache lives in the CPU-side RSS that was measured
growing 35 -> 47.6 GB. Also drop `--cache-reuse 256`, which this context ignores.

## Scope addition (received mid-run): a fourth arm, `gguf8`

After gguf6, run `gguf8` = `~/Models/qwen3.8-27b-gguf/Qwen3.8-27B-Q8_0.gguf` (~29 GB).
The alias is already wired in `llm-serve` (line 103) and the download started 15:14.
The deliverable for this arm is a **(`-c` setting x depth) frontier**, not a single
ladder: if the server fails to load or OOMs at `-c 262144`, retry at `LLM_CTX=131072`
then `65536`, recording the largest `-c` that loads and runs, with a Phase A warmth
check at whatever `-c` fits.

Predicted wired budget (using the 64 KB/token KV measured above):

| `-c` | KV GiB | weights GiB | model total GB | + 3.1 GB system | vs 50 GB limit |
|---|---|---|---|---|---|
| 262,144 | 16.0 | 27.0 | 46.2 | **49.3** | **~0.7 GB spare — expected to be marginal** |
| 131,072 | 8.0 | 27.0 | 37.6 | 40.7 | 9.3 GB spare |
| 65,536 | 4.0 | 27.0 | 33.3 | 36.4 | 13.6 GB spare |

Two levers if `-c 262144` will not load. `LLM_MLOCK=0` (llm-serve line 387) unpins the
weights — llm-serve's own comment warns prefill collapsed 196 -> 4 tok/s without mlock
when the machine is under memory pressure, so that trade must be *measured*, not assumed.
`-cram` also matters here: the prompt cache is host RAM on top of everything above, and
at 103 KB/token a single deep state is ~10-25 GB.

---

## Runner handover — 15:24, campaign resumed

The previous runner was stopped externally. State reconciled at 15:24:

- `sysctl kern.boottime` = `{ sec = 1787982452 } Fri Aug 28 22:47:32 2026` — **identical to the
  campaign-start value**, and the panic-file count is still **1**. No machine restart, no panic:
  everything below is continuous with everything above.
- llama-server was still up on gguf4 (port 8089, RSS 37.2 GB, health ok), and the previous
  runner's background ladder driver (`driver.sh gguf4 1 10`, 29-min per-turn watchdog) was
  **still alive and mid-flight on turn B2**. It was left to finish rather than restarted.
- A `memsample.sh` sampler was also still writing `gguf-turns/mem-gguf4.csv` at 5 s intervals.
- All four GGUF files are present and size-stable, `gguf8` (`Qwen3.8-27B-Q8_0.gguf`,
  29,047,086,048 B) included — its download completed at 15:21.

A supervisor (`campaign.sh`) now chains the remainder: cap the gguf4 ladder, then gguf5,
gguf6, and the gguf8 `-c` frontier, stopping the stack at the end.

### Ladder scope, and why it is capped

The gguf4 ladder is capped at **B4** (or B3 if B3 fails), and the gguf5/gguf6/gguf8 ladders at
**B3**. This is a deliberate consequence of the cost structure established by B1/B2 rather than
a shortcut: each ladder turn's cost is dominated by the *fallback re-prefill from the 18,339-token
system-prompt boundary* described above, so turn N costs roughly
`(46,000 x N - 18,339) / ~130 tok/s` seconds of prefill. That is ~5 min at B1, ~10 min at B2,
~20 min at B3 and ~30 min at B4 — the 29-min watchdog becomes the binding constraint before
memory ever does. **On llama.cpp the ceiling is a time wall, not the OOM wall that ended the MLX
arms.** Depths past B4 would only re-measure the same wall more expensively.

### One knob deliberately left at its default

`--cache-ram` stays at the 8192 MiB default for every arm. `llm-serve` (line 590) hardcodes
`--cache-reuse 256` and passes no `-cram`, so 8192 MiB is what a user of this stack actually gets;
measuring it is the point. The `-cram` fix is recorded above as an actionable, not applied
mid-campaign.

### gguf4 ladder, continued

| turn | file | status | wall s | reqs | **ctx depth reached** | cached frac | prefill s | recall | RSS GB | wired GB | free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | file1 | OK | 312 | 3 | **65,146** | 0.637 | 294 | **yes** (TANGERINE-LYNX) | 45.15 | 40.21 | 10.80 |
| B2 | file2 | OK | 844 | 3 | **110,208** | 0.656 | 817 | **yes** (COBALT-HERON) | 34.73 | 40.37 | 4.77 |

**B2 is already the headline number for the depth question: a correct, complete Claude Code
turn at 110,208 tokens of context.** MLX's best arm (mlx4) OOM'd at 80,625 and failed totally
at 119,848; llama.cpp sailed through 110k with the answer right and never touched the wired
limit — wired held at 40.37 GB, 9.6 GB *under* the 50 GB cap, at 110k depth.

The per-request breakdown for B2 shows the campaign's whole cost model in three lines:

```
req1  prompt= 65,170  cached=65,109 (100%)  f_keep=0.999  reproc=    61  ->    1 s
req2  prompt= 91,772  cached=18,338 ( 20%)  f_keep=0.289  reproc=73,434  ->  577 s
req3  prompt=110,165  cached=91,770 ( 83%)  f_keep=0.998  reproc=18,395  ->  239 s
```

Requests 1 and 3 continue the live slot almost perfectly. Request 2 — the one that appends the
big Read tool result — falls back to the same **18,338-token** system-prompt boundary seen at
every depth and re-prefills 73,434 tokens. That single request is 577 of the turn's 844 seconds,
i.e. **68% of the wall clock of a deep turn is one cache miss.** Fixing `-cram` targets exactly
this and nothing else.

**Free memory is the metric that moved**, not wired: 10.80 GB at 65k -> **4.77 GB** at 110k.
Wired is pinned by the preallocated KV and does not budge; the prompt cache and page-in pressure
eat host RAM instead. On llama.cpp, `free` is the early-warning signal that `wired` was on MLX.

Also note RSS *fell* 45.15 -> 34.73 GB between B1 and B2. That is not a leak fix, it is the
`--cache-ram` eviction (`removing oldest entry`) discarding the deep slot state — the same event
that causes the f_keep=0.289 collapse. **RSS dropping is the symptom of the cache miss that is
about to cost 577 seconds.**

### B3 — the arithmetic ceiling arrives exactly where predicted

| turn | file | status | wall s | reqs | ctx depth reached | cached frac | prefill s | recall | RSS GB | wired GB | free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B3 | file3 | **watchdog kill @ 29 min** | 1741 (rc=137) | 3 done + 1 cut | **110,809** | 0.666 | 892 | **no** (empty) | 37.46 | 39.55 | 5.82 |

```
req1  prompt=110,205  cached=      1 (  0%)  f_keep=0.147  reproc=110,204  ->  886 s
req2  prompt=110,408  cached=110,200 (100%)  f_keep=0.999  reproc=    208  ->    3 s
req3  prompt=110,650  cached=110,404 (100%)  f_keep=0.998  reproc=    246  ->    3 s
req4  (task 1938)  re-prefilling ~138k, 74% done at 102,168 tok when the watchdog fired
```

**`cached = 1 token`.** Not a degraded hit — a total miss. The turn opened by re-prefilling the
entire 110,204-token conversation from zero, at 124 t/s, for **886 seconds**, before it could say
anything. Requests 2 and 3 were then instant (3 s each, f_keep≈0.999): once the slot is live it
stays live. The cost is entirely in *re-entering* the conversation.

This is the `-cram` arithmetic closing exactly as derived earlier:

> 8,192 MiB / 103 KB per token = **~79,000 tokens**

B2 ended at 110,208 tokens. A 110k slot state costs ~11.3 GB, which **cannot fit** in the 8 GiB
prompt cache at any price — the server discards it (`prompt state size ... exceeds cache size
limit ..., skipping`), so when B3 came back to the same session there was nothing to restore and
nothing to match. B2's 110k success and B3's total miss are the two sides of one boundary.

**Revised ceiling statement for gguf4.** Memory never bound: wired sat at 39.55 GB and free at
5.82 GB, with 10 GB of wired headroom unused at the deepest point measured. What bound the run is
that **past ~79k tokens every new turn costs a full from-zero re-prefill of the whole
conversation** — ~15 min at 110k, ~21 min at 138k, and rising linearly. The gguf4 ceiling is a
**cache-configuration ceiling, not a memory ceiling**, and it is a one-line fix (`-cram`).

The B3 kill is my own 29-minute watchdog, not an engine failure — the turn was still making
forward progress when it was cut. Read the row as "a 138k-token turn needs more than 29 minutes
of wall clock under the default `-cram`", not as a crash. The watchdog was raised to **45 min**
for the gguf5/gguf6/gguf8 arms so their deep turns are not truncated the same way.

### ARM 1 (gguf4) — summary

| metric | result |
|---|---|
| deepest **correct** turn | **110,208 tokens** (B2, COBALT-HERON, 844 s) |
| deepest context reached | 110,809 tokens |
| ceiling type | **prompt-cache (`-cram` 8 GiB), not memory** |
| wired at every depth | flat **39.6–40.4 GB** (preallocated 262k KV), 10 GB spare vs the 50 GB cap |
| free RAM at depth | 10.80 GB @65k -> 4.77 GB @110k |
| Phase A warm hits | **11 / 11**, 8–17 s per follow-up turn |
| decode t/s | 17.4 @19k -> 13.0 @46k -> 12.2 @65k -> 9.2 @110k |
| prefill t/s | 295 @19k -> 188 @46k -> 132 @65k -> 127 @92k -> 124 @110k |
| MTP draft acceptance | 0.62–0.96, mean length 2.4–2.6 |
| failures | 1 (B3, watchdog, not the engine) |
| panics | 0 |

---

# ARM 2 — gguf5 (Qwen3.8-27B-UD-Q5_K_XL, 20.9 GB file)

Server start 15:55, `-c 262144`, `--mlock`. **Load to health: 10 s.** Same two startup
lines as gguf4 — the inline MTP draft context is created, and `cache_reuse is not supported
by this context, it will be disabled`. `n_slots = 1, n_ctx_slot = 262144, kv_unified = 'false'`.

## Phase A — warmth (gguf5)

Session `56ae0707-…`, same six questions as gguf4.

| turn | wall s | reqs | max ctx | cached frac | prefill t/s | decode t/s | draft acc | verdict |
|---|---|---|---|---|---|---|---|---|
| A1 | 161 | 3 | 31,714 | 0.543 | 293 / 156 / 212 | 14.5 / 15.2 / 12.5 | 0.84 / 0.90 / 0.69 | cold (first ever) |
| A2 | **33** | 1 | 32,027 | **0.996** | 93 | 14.5 | 0.875 | **WARM** |
| A3 | **16** | 1 | 32,169 | **0.987** | 143 | 13.3 | 0.782 | **WARM** |
| A4 | **12** | 1 | 32,306 | **0.994** | 116 | 13.5 | 0.802 | **WARM** |
| A5 | 69 | 3 | 34,257 | **0.984** | 108 / 142 / 178 | 14.2 / 15.2 / 12.0 | 0.85 / 0.96 / 0.67 | **WARM** |
| A6 | 85 | 3 | 36,031 | **0.984** | 148 / 120 / 166 | 12.8 / 13.9 / 11.7 | 0.73 / 0.84 / 0.65 | **WARM** |

All six answered correctly. A6 even corrected the question's premise (there are three MLX patch
scripts in `scripts/`, not two) — worth noting because it is a quality signal the shallower
quants can be checked against.

**Warm-cache verdict: 12 / 12 continuation requests warm, mean cached fraction 0.989.** The
build-10621 fix is not quant-specific — Q5_K_XL behaves exactly like Q4_K_XL. Follow-up turns
cost **12–33 s**.

Idle-after-load footprint: **RSS 38.4 GB, wired 42.86 GB** (gguf4: 35.0 / 39.61). The
**+3.25 GB of wired is almost exactly the +3.3 GB weight-file delta** (20.9 vs 17.6 GB), which
confirms the model: KV is fixed by `-c`, and quant choice moves wired one-for-one with file size.

### Calibrated wired-memory model (two measured points, two falsifiable predictions)

gguf4 and gguf5 give two points on the same line, because `-c` was identical (262,144) for both:

| arm | weights file GB | measured idle wired GB | file + 22.05 |
|---|---|---|---|
| gguf4 | 17.56 | **39.61** | 39.61 |
| gguf5 | 20.88 | **42.86** | 42.93 |

Delta file = 3.32 GB, delta wired = 3.25 GB — **slope 0.98, i.e. one-for-one.** So

> **idle wired GB ≈ weights_file_GB + KV_GB + 4.87**, with KV_GB = 65,536 B/token x `-c`
> (17.18 GB at `-c 262144`), and 4.87 GB = 3.10 GB system baseline + ~1.77 GB of SSM/conv
> state, compute buffers and the MTP draft context.

Both measured points fit to within 70 MB. Applying it forward, **before running either arm**:

| arm | weights GB | `-c` | KV GB | predicted idle wired GB | vs 50 GB cap |
|---|---|---|---|---|---|
| gguf6 | 25.30 | 262,144 | 17.18 | **47.35** | 2.65 GB spare — should load, tight |
| **gguf8** | **29.05** | **262,144** | **17.18** | **51.10** | **1.1 GB OVER the cap — predicted to fail** |
| gguf8 | 29.05 | 131,072 | 8.59 | **42.51** | 7.5 GB spare |
| gguf8 | 29.05 | 65,536 | 4.29 | **38.21** | 11.8 GB spare |

This sharpens the earlier estimate at the end of the pre-handover plan, which put gguf8 at
`-c 262144` at 49.3 GB and "expected to be marginal". That estimate used the raw 27.0 GiB
weights figure and omitted the ~1.8 GB of non-KV overhead now measured twice. **The corrected
prediction is that gguf8 at `-c 262144` does not fit, and that `-c 131072` is the frontier.**
The campaign script tries 262,144 first anyway, so the prediction gets tested rather than assumed.

## Phase B — depth ladder (gguf5)

Fresh session `c6d5fc8b-…`, `LLM_CTX=262144`, per-turn watchdog raised to 45 min.

| turn | file | status | wall s | reqs | ctx depth reached | cached frac | prefill s | recall | RSS GB | wired GB | free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B1 | file1 | OK | 462 | 7 | **67,259** | 0.674 | 402 | **yes** (TANGERINE-LYNX) | 45.27 | 42.82 | 9.53 |

gguf5's B1 took 7 requests where gguf4's took 3 — the model chose to issue more intermediate
tool calls — so the two are not wall-clock comparable turn-for-turn. The per-request structure is
the identical shape, though, and it reproduces the gguf4 finding exactly: **five cheap requests
(1–7 s each, f_keep 0.99+) and one expensive one.** Request 6, the one that appends the big Read
result, dropped to `f_keep=0.883` / 38% cached and re-prefilled 30,170 tokens for 171 s; request 7
then cost another 156 s to climb to 67k. **327 of the turn's 402 prefill seconds are those two
requests.**

Note request 6 fell back to `cached = 18,153` — the *same* ~18.2k system-prompt boundary as
gguf4's 18,338/18,339. The boundary is a property of the Claude Code prompt layout, not of the
quant.

| turn | file | status | wall s | reqs | ctx depth reached | cached frac | prefill s | recall | RSS GB | wired GB | free GB |
|---|---|---|---|---|---|---|---|---|---|---|---|
| B2 | file2 | OK | 874 | 3 | **112,304** | 0.656 | 848 | **yes** (COBALT-HERON) | 43.73 | 42.85 | 6.97 |
| B3 | file3 | OK | 1533 | 3 | **158,766** | 0.658 | 1508 | **yes** (SAFFRON-MARMOT) | 40.70 | 42.72 | 5.18 |

**gguf5 completed all three ladder turns correctly and reached 158,766 tokens.** That is the
deepest correct Claude Code turn measured anywhere in either campaign — **1.97x mlx4's OOM onset
(80,625) and 1.32x mlx4's total-failure depth (119,848)** — on the *larger* of the two quants
tested so far, with wired never moving off 42.7–42.9 GB.

### Why gguf4 died at B3 and gguf5 did not — resolved

This looked like a contradiction (gguf4 collapsed re-entering a 110,208-token session; gguf5
sailed into a 112,304-token one) so it was traced in the logs. Both arms hit the *same* wall:

```
gguf4 B3:  W srv alloc: - prompt state size  9769.500 MiB exceeds cache size limit 8192.000 MiB, skipping
gguf5 B3:  W srv alloc: - prompt state size 11123.626 MiB exceeds cache size limit 8192.000 MiB, skipping
```

So on **both** quants the prompt cache is dead past ~79k — no deep state is ever saved, and
restoring one is impossible. What differs is only whether the **live slot** still prefix-matched
the next request:

| | gguf4 B3 req1 | gguf5 B3 req1 |
|---|---|---|
| `f_sim_best` | **0.147** | **0.999** |
| tokens reused | **1** | 112,227 |
| prefill | **886 s** | **2 s** |

And the reason 0.147 is fatal rather than merely bad is the hybrid architecture: with
GatedDeltaNet layers you cannot rewind a slot to an arbitrary position, only to a stored
**checkpoint**. There is effectively one usable checkpoint, at the ~18.2k system-prompt boundary.
Every partial hit in this campaign falls back to exactly that number — gguf4 kept 18,338 / 18,339
/ 18,395; gguf5 kept 18,151 / 18,152 / 18,153. gguf4's B3 matched only 0.147 x 110,205 ≈ **16,200
tokens, i.e. *below* the 18.2k checkpoint**, so there was no checkpoint at or under the match
point and the only legal action was a full re-prefill from zero.

**The practical statement: past ~79k the run is a coin flip.** The prompt cache is guaranteed
useless, so a deep session survives only while its live slot keeps matching. When Claude Code
re-emits the conversation in a way that perturbs anything before ~18.2k, the cost is not a slow
turn but a complete from-zero re-prefill — 886 s at 110k, and linear in depth after that. This is
not quant-dependent; gguf4 lost the coin flip and gguf5 won it.

### ARM 2 (gguf5) — summary

| metric | result |
|---|---|
| deepest **correct** turn | **158,766 tokens** (B3, SAFFRON-MARMOT, 1533 s) |
| ladder result | **3 / 3 turns correct** |
| ceiling type | not reached — no memory ceiling, no failure |
| idle wired / at depth | 42.86 GB / flat **42.7–42.9 GB** (7 GB spare vs the 50 GB cap) |
| free RAM at depth | 9.53 GB @67k -> 6.97 GB @112k -> 5.18 GB @159k |
| Phase A warm hits | **12 / 12**, 12–33 s per follow-up |
| decode t/s | 15.8 @19k -> 11.6 @48k -> 11.0 @94k -> 9.3 @112k -> **7.8 @159k** |
| prefill t/s | 291 @19k -> 176 @48k -> 124 @94k -> 102 @140k -> 61 @159k |
| MTP draft acceptance | 0.68–0.98 |
| failures / panics | 0 / 0 |

---

# ARM 3 — gguf6 (Qwen3.8-27B-UD-Q6_K_XL, 25.3 GB file)

Server start 16:51, `-c 262144`, `--mlock`. Loads healthy. Idle-after-load:
**RSS 42.49 GB, wired 46.86 GB, free 8.36 GB.**

### Prediction check #1

The calibrated model predicted **47.35 GB**; measured **46.86 GB** — **error -0.49 GB (1.0%)**.
The model is sound but the non-KV constant drifts slightly downward with quant size:

| arm | file GB | measured wired GB | implied constant (wired - file - 17.18) |
|---|---|---|---|
| gguf4 | 17.56 | 39.61 | 4.87 |
| gguf5 | 20.88 | 42.86 | 4.80 |
| gguf6 | 25.30 | 46.86 | **4.38** |

Re-running the gguf8 prediction across all three fits (constant 4.87 / 4.80 / 4.38, and a
trend-extrapolated ~4.1) gives **50.3–51.1 GB** for gguf8 at `-c 262144`. **Every fit still puts
it over the 50 GB `iogpu.wired_limit_mb`**, so the prediction stands, though the margin is only
0.3–1.1 GB and the limit is not a hard allocation failure — gguf6 at 46.86 GB leaves just 3.1 GB
of headroom and is already the tightest configuration that has loaded cleanly.
