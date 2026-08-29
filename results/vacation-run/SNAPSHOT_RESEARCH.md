# APC snapshot research — why the exact-mode store copies the world, and what to do about it

Date: 2026-08-29 · Target: Qwen3.8-27B (hybrid GatedDeltaNet + 16 full-attn layers), M5 Pro 64 GB,
`mlx_vlm.server` 0.6.17, exact-mode APC.

Confidence markers: **[code]** = confirmed by reading the installed source · **[upstream]** = confirmed
against a primary GitHub source (issue/PR/API state) · **[inferred]** = reasoned, not measured.

Local package root in all `file:line` citations:
`/Users/yash/Desktop/Programming/local_llm_bench/.mlxenv-0617/lib/python3.12/site-packages/mlx_vlm/`

---

## 0. Headline

The store is a deep copy because **the live KV cache is a preallocated buffer written by in-place index
assignment**, so a snapshot that merely referenced it would be scribbled on by the next decode step. That
part is unavoidable in the current cache design. But three things on top of it are *not* necessary and are
what actually kills us:

1. **0.6.17 clones the whole cache twice per store** (`ar.py:1872` clones, then `apc.py:3146` clones the
   clone). Peak is ~3× the full-attention KV, not 2×. Upstream PR #2072 (opened 2026-08-28, still open)
   fixes exactly this with `take_ownership=True`. **[code] [upstream]**
2. The clone is emitted as **one `mx.eval` over every layer at once** (`apc.py:321`), i.e. a single Metal
   command buffer holding ~2.9 GB of copies. **[code]**
3. **The disk tier does not avoid any of it** — the GPU clone happens *before* the disk branch and the
   disk writer re-evaluates slices of the clone. **[code]**

And the checkpoint is nearly the whole prompt by design: `checkpoint_len = len(prompt) - 16`
(`APC_EXACT_PREFIX_GUARD_TOKENS`, `apc.py:2880`, used at `ar.py:2363`). There is no "small checkpoint"
knob — the guard interval is a *tail* guard, not a spacing interval. **[code]**

---

## 1. Why must the store be a deep copy at all?

### 1.1 The live cache mutates in place — confirmed

`models/cache.py:345-367` (`KVCache.update_and_fetch`):

```python
        self.offset += keys.shape[2]
        self.keys[...,   prev : self.offset, :] = keys      # cache.py:365
        self.values[..., prev : self.offset, :] = values    # cache.py:366
        return self.keys[..., : self.offset, :], self.values[..., : self.offset, :]
```

The buffer is preallocated in `step`-sized chunks (`step = 256`, `cache.py:338`) and grown by
`mx.concatenate` only when capacity runs out. Between growths, every prefill chunk and every decode step
writes into the *same* array object by index assignment.

MLX arrays are immutable *values*, but `mx.array.__setitem__` is an in-place update of the array object's
descriptor: the Python object keeps its identity and any graph node that captured it as an input resolves
against the post-write contents. So a "zero-copy snapshot" that stored `c.keys` (or an un-evaluated slice
of it) would, on evaluation, see whatever the live cache had written since. **[code] + [inferred]** for
the MLX descriptor semantics — the code's behaviour (eval-then-copy in every adapter) is only consistent
with that reading.

Hence `_copy_mlx_array` (`apc.py:262-264`):

```python
def _copy_mlx_array(x: mx.array) -> mx.array:
    """Materialize ``x`` into a fresh MLX-owned contiguous buffer."""
    return mx.contiguous(mx.array(x, dtype=x.dtype))
```

and `KVCacheCloneAdapter.clone` (`apc_adapters.py:322-345`) which does
`copy(c.keys[..., :off, :])` per layer, appends both tensors to `eval_targets`, and
`_clone_prompt_cache_for_apc` (`apc.py:304-322`) finishes with **one** `mx.eval(eval_targets)` over all
32 tensors (16 layers × K,V). **[code]**

### 1.2 Rollback is the second reason, and it is real

`KVCache.trim` (`cache.py:390-393`) only decrements `offset` — it does not clear data. A subsequent
`update_and_fetch` therefore *rewrites* the region that was trimmed away. So any snapshot that aliased
the live buffer would be silently corrupted by an MTP verify-round rejection or a continuous-batching
rollback that trims below the snapshot's length and regenerates. **[code]**

Note the asymmetry that matters for §2: **above** the trim watermark the buffer is rewritten, **below**
it, KV for token *i* is write-once. A delta scheme is therefore valid for any region below a committed
watermark.

### 1.3 The recurrent layers genuinely cannot be anything but a copy

`ArraysCache` (`cache.py:626+`) holds a fixed-size list of state tensors with no per-token history;
`ArraysCacheCloneAdapter` (`apc_adapters.py:385-410`) copies each state wholesale. There is no slicing,
no trimming, no reconstruction from blocks. This is why the model is routed to exact mode at all:
`model_apc_mode` (`apc.py:4154-4173`) returns `"block"` only if **every** entry is block-eligible, and
`apc_block_eligible` (`apc_adapters.py:280-285`) is a strict type-table check that `ArraysCache` fails.
One non-pageable layer forces the *whole model* onto the whole-prefix snapshot path. **[code]**

### 1.4 The part that is *not* necessary: the store clones twice

`ar.py:1858-1880`:

```python
    def _apc_prompt_cache_for_store(self, batch_idx: int) -> Optional[List[Any]]:
        return _apc.snapshot_prompt_cache_row(self.prompt_cache, batch_idx)   # ar.py:1859
...
            prompt_cache = self._apc_prompt_cache_for_store(batch_idx)        # ar.py:1872
            self._apc_manager.store_exact_cache(                              # ar.py:1875
                meta["full_input_ids"][:checkpoint_len], prompt_cache, ...)
```

`snapshot_prompt_cache_row` (`apc.py:4007-4028`) ends in `_clone_prompt_cache_for_apc(...)` — **clone #1,
fully evaluated**. `store_exact_cache` then does, unconditionally (`apc.py:3146`):

```python
        copied = _clone_prompt_cache_for_apc(prompt_cache)                    # apc.py:3146
```

— **clone #2, fully evaluated**. Same second store site at `ar.py:2115-2118`. **[code]**

Peak accounting at 45k tokens (≈2.9 GB of full-attention KV, plus ~72 MB recurrent):

| resident at the moment of clone #2 | bytes |
|---|---|
| live cache (step-padded, ≥ prefix) | ~2.9 GB |
| clone #1 (held by `ar.py` local) | ~2.9 GB |
| clone #2 (being materialized) | up to ~2.9 GB |
| previously retained checkpoint entry (`ENTRIES=2`) | ~2.9 GB |

≈ **8.7 GB transient + 2.9 GB retained**, in a single Metal command buffer for the last term. That is the
OOM/panic budget, and roughly a third of it is pure redundancy.

Upstream fix, still open: **Blaizzy/mlx-vlm PR #2072** "Reduce exact APC ownership-transfer peaks"
(opened 2026-08-28), which adds `take_ownership: bool = False` to `store_exact_cache` and skips clone #2:
> "The store side had the inverse ownership issue: the coordinator first created a detached checkpoint
> snapshot, then `APCManager.store_exact_cache()` cloned the same snapshot again."

Its diff touches `apc.py`, `apc_adapters.py`, `apc_coordinator.py`, `generate/ar.py` and adds
`test_exact_cache_store_can_take_ownership_without_cloning`. **[upstream]**
Note it is written against post-#1960 `main` which has an `apc_coordinator.py`; **0.6.17 has no
`apc_coordinator.py`, no `take_ownership`, and no `APC_EXACT_MAX_PROMPT_TOKENS`** (grep: zero hits across
the installed package). The PR does not apply as-is; the one-line idea does. **[code] [upstream]**

**Answer to Q1:** yes, a deep copy is required by the current buffer design (in-place index assignment +
trim-then-rewrite). No, the *number* of deep copies is not required — we are paying for two, and only
because 0.6.17 lacks an ownership contract.

---

## 2. Could the store be incremental / delta?

### 2.1 Semantically, yes — and the shape of it is known

Split the store by layer type:

* **16 full-attention layers** — KV for token *i* is write-once below the commit watermark (§1.2), so
  storing only `[last_checkpoint_len : new_checkpoint_len)` and chaining to the previous entry is sound.
  Cost per store: `new_tokens × 64 KB` instead of `total_tokens × 64 KB`.
* **GatedDeltaNet layers** — fixed-size `ArraysCache`, must be copied whole, at the exact boundary. ~72 MB.

Steady-state store for a 1,200-token turn: ~72 MB + ~77 MB ≈ **0.15 GB instead of 2.9 GB**, a ~19×
reduction in per-store churn. **[inferred]**

The blocking constraint is that **the recurrent state is only valid at the exact position it was saved
at**, so restore granularity is quantized to checkpoint boundaries even though the attention part is
fine-grained. Every production implementation that solved this landed on the same shape.

### 2.2 mlx_vlm already has both halves — it just cannot mix them

`store_kv_blocks` (`apc.py:3321`) + the layer-major disk block store (`apc.py:2423`,
`load_layer_major_prefix` at `apc.py:2091`) are a real incremental, chunk-evaluated, block-hashed path —
with `APC_DISK_EVAL_BLOCK_CHUNK` (`apc.py:2904`) and `APC_DISK_LOAD_BLOCK_CHUNK` (`apc.py:2913`) already
chunking `mx.eval` on the *load* side. It is unreachable for us because `model_apc_mode` is all-or-nothing
(`apc.py:4168-4171`). A **per-layer hybrid mode** — blocks for the `KVCache` layers, an exact checkpoint
for the `ArraysCache` layers, intersected at a common boundary — is the missing piece, and it is entirely
inside `apc.py`/`apc_adapters.py`. No upstream issue proposes it yet. **[code]**

### 2.3 What other implementations do

**vLLM — has exactly this, and it is the reference design. [upstream]**
- `--mamba-cache-mode align`: attention KV is paged (blocks are the storage, so "storing" costs nothing —
  no copy at all), and the Mamba/GDN state snapshot is retained **only at block boundaries**, with the
  attention block size inflated to the Mamba page size (528 tokens for Qwen3.5-class, up to 2096 for
  larger members).
- [PR #46384](https://github.com/vllm-project/vllm/pull/46384) "[2/N][Core] support partial prefix cache
  hit for hybrid model" (merged 2026-06-22) adds `prefix_match_unit` to decouple match granularity from
  physical block size, "scheduler boundaries needed to materialize and resume partial Mamba states",
  copy-on-write for continuing requests, per RFC #45702. Usage:
  `vllm serve Qwen/Qwen3.5-35B-A3B-FP8 --enable-prefix-caching --mamba-cache-mode align --prefix-match-unit 16`.
- The known failure mode is instructive for us:
  [issue #45238](https://github.com/vllm-project/vllm/issues/45238) (open) — only the checkpoint at the
  *last block boundary before the prompt ends* is retained, so if that position lands in request-unique
  tokens, the Mamba group misses, `HybridKVCacheCoordinator` intersects per-group hits, and **all** reuse
  drops to 0% even though the attention blocks matched perfectly. That is the same "one checkpoint at
  n−16" fragility we have, only with better metrics.
- Also relevant: [#40696](https://github.com/vllm-project/vllm/issues/40696) (prefix caching ineffective
  when prompt < block_size), [#37898](https://github.com/vllm-project/vllm/pull/37898) Marconi-style
  admission policy for hybrid cache.

**LM Studio `mlx-engine` — checkpoint-based, but still a full deep copy per store. [upstream]**
`mlx_engine/cache_wrapper.py` (fetched from `main`, 336 lines): `CacheWrapper._store_snapshot` at
line 97-111 does literally `copy.deepcopy(cache)` (line 109) into an `LRUPromptCache` (line 85-88); the
checkpoint position is `checkpoint_prefix_len = total_prompt_tokens - self._checkpoint_tail_tokens`
(line 272) — i.e. the *same* "whole prompt minus a tail guard" design as mlx_vlm's
`len(ids) - exact_cache_guard_tokens`. `_restore_cache` (line 141-176) prefers trimming and falls back to
checkpoints "since some KV caches are non-trimmable". **So LM Studio has no incremental store either** —
it has the same O(context) copy, just with `copy.deepcopy` instead of `mx.contiguous`. Their open tracker
items confirm the same pain: `lmstudio-ai/mlx-engine#327` (kv cache restarts from 0 every other turn),
`#177` (RotatingKVCache trim erases whole cache), `lmstudio-bug-tracker#1818`, `#1862`.

**mlx-lm — no incremental snapshot, and the systemic gap is documented. [upstream]**
[ml-explore/mlx-lm#980](https://github.com/ml-explore/mlx-lm/issues/980) "Prefix cache reuse is broken for
all hybrid-architecture models (sliding window, SSM/Mamba)" (closed 2026-03-11) is the unifying issue —
`can_trim_prompt_cache` is False for any cache list containing `ArraysCache`, so reuse falls back to full
recompute. `make_prompt_cache`/`save_prompt_cache`/`trim_prompt_cache` have no delta or zero-copy path.
Separately, [#1480](https://github.com/ml-explore/mlx-lm/issues/1480) (open, 2026-07-06) reports Metal
command-buffer OOM during long-context prefill on Qwen3.6-35B-A3B "despite only 10 KV-cache layers",
attributing it to *transient* Metal allocation rather than final KV size — the same class of failure we
see, and evidence that big single-command-buffer transients are the trigger.

**mlx-vlm — the memory problem is known upstream, with two open PRs. [upstream]**
- [#1576](https://github.com/Blaizzy/mlx-vlm/pull/1576) (open, 2026-07-11) `APC_EXACT_MAX_PROMPT_TOKENS`.
  Its problem statement is our bug verbatim: *"APC exact mode snapshots the entire prompt KV of every
  completed prefill, and the exact cache is LRU by entry count, not by size… a single cache entry can
  weigh anywhere from tens of MB to ~2.6 GB… The exact store deep-copies the row's prompt cache on the
  generation thread; at doc scale that is a multi-GB copy per completed prefill… Thread dumps under
  concurrent long-prompt load repeatedly catch the generation thread inside `_apc_prompt_cache_for_store`."*
  It documents the same `libc++abi … [METAL] Command buffer execution failed: Insufficient Memory` abort.
  The mitigation it offers is a *cap* (skip the store above N tokens), not a delta store.
- [#2072](https://github.com/Blaizzy/mlx-vlm/pull/2072) (open, 2026-08-28) the double-clone fix, §1.4.
- Adjacent: [#1835](https://github.com/Blaizzy/mlx-vlm/pull/1835) (decline prefix reuse for non-trimmable
  recurrent caches — the `ArraysCache has no attribute 'trim'` crash),
  [#2048](https://github.com/Blaizzy/mlx-vlm/issues/2048) (exact-mode APC silently restores 0 tokens for
  hybrid caches with integer state / KVCache subclasses — worth checking our stats for
  `matched_tokens >> served_tokens`), [#999](https://github.com/Blaizzy/mlx-vlm/issues/999) (server clears
  Metal cache after every request).
- **No upstream issue or PR proposes a chunked `mx.eval` in the clone path, or a delta/incremental exact
  store.** That idea appears to be unclaimed.

### 2.4 Why our append-only hack diverged

`bench_append_only.py` + `results/append_only.jsonl`: 27B `speedup 119.3`, `output_preserved false`;
35B-A3B `speedup 87.4`, `output_preserved true` — *identical code path, identical prompt, same cache kinds
`["ArraysCache","KVCache"]`*. The mechanism is therefore sound; the divergence is not a reuse bug.

Most likely cause **[inferred]**: the warm arm prefills the 21-token suffix in one shot while the cold arm
prefills 23,116 tokens in `PREFILL_STEP_SIZE`-sized chunks (`config.PREFILL_STEP_SIZE`, pinned at 2048 per
`REPORT.md:279`). Different chunk boundaries change the reduction/accumulation order — benign for
attention (recomputed exactly from K/V) but **not** for GatedDeltaNet, whose state is a sequential
chunkwise scan, so the recurrent state differs in the low bits. At `temperature 0.0` a single near-tie in
the argmax flips one token and the continuation diverges thereafter. The 35B-A3B matching byte-for-byte
is consistent with "no near-tie happened to be crossed", not with "the 35B is exact and the 27B isn't".

A secondary candidate worth one experiment: an off-by-one at the seam — the cache after turn 1 holds
`ids1 + gen1[:-1]` (the final sampled token is never fed back) while the cold arm's `full2` includes all
of `gen1`. If so the warm arm is conditioning on a one-token-shorter context. Cheap to test: compare
`cache[0].offset` against `len(ids1) + len(gen1)` before building `full2`. **[inferred]**

Either way: **the divergence is a fidelity question about the append-only shortcut, not evidence against
incremental storage.** A delta store re-uses the *same* state the live cache produced, so it has no
chunk-boundary discrepancy at all.

---

## 3. GGUF / llama.cpp status right now (2026-08-29)

Verified via the GitHub API today, not from memory.

| ref | title | state |
|---|---|---|
| [#19794](https://github.com/ggml-org/llama.cpp/issues/19794) | Qwen3-Coder-Next hybrid prompt cache forces full re-processing despite `--swa-full` | **closed as not planned** 2026-05-15 |
| [#21831](https://github.com/ggml-org/llama.cpp/issues/21831) | Server forces full prompt re-processing on subsequent requests (SWA/recurrent) | **open** (52 comments) |
| [#22384](https://github.com/ggml-org/llama.cpp/issues/22384) | server: fix context checkpoint restore for hybrid/recurrent (DeltaNet/Mamba) | **closed 19 minutes after opening**, 2026-04-26 — self-closed, points at a *fork* (`spiritbuun/buun-llama-cpp#26`). **Not an upstream fix.** |
| [#24055](https://github.com/ggml-org/llama.cpp/issues/24055) | Context checkpoints always invalidated on hybrid/recurrent models | **open** (19 comments), the live thread |

Merge status of every PR named in that thread (API-verified):

| PR | title | merged |
|---|---|---|
| #22929 | server: fix checkpoints creation | **merged 2026-05-25** — *this is the regression*: it moved checkpoint creation to the last user message only |
| #24411 | server : skip checkpoints beyond `pos_next` | merged 2026-06-11 |
| #24176 | server: create checkpoints at every user message | merged 2026-06-23 |
| #24797 | server: fix checkpoint handling for hybrid memory models | **closed, never merged** (rejected: could reuse recurrent state at positions it was never valid for) |
| #25472 | server : evict checkpoints within min-step of each other | merged 2026-07-12 |
| **#25592** | **server: fix checkpoint handling for hybrid/recurrent models (#24055)** | **OPEN** — created 2026-07-12, last updated 2026-08-26, *still not merged as of today* |

**#25592 is the fix for our exact failure.** Its own description:
> "checkpoint metadata records the actual validity at save time (`pos_min = pos_max` for hybrid/recurrent)
> instead of relying on what `seq_pos_min` happens to report… a checkpoint is only restored when its exact
> position lies inside the common prefix of the new prompt… On top of that, the cache is made useful for
> agentic clients, **which strip the reasoning of the previous reply so the next request diverges right at
> the end of the previous prompt**: end-of-prompt checkpoints are exempt from `--checkpoint-min-step`…"

Thread status as of 2026-08-18 (two participants, one correcting the other):
> **#25472** — merged, in current builds → covers the bounded single-ubatch case
> **#25592** — still open, awaiting review → the full-reprocess case reported in this issue is **not**
> fixed in release builds yet.

### 3.1 Our build 9620 was two months stale and sat in the worst window

`b9620` was published **2026-06-13** (GitHub releases API, commit `57fe1f07`). So the Aug 22 Trial A ran on
a build that:
- **contained** the #22929 regression (2026-05-25) — checkpoints only at the last user message;
- **predated** #24176 (Jun 23), #25472 (Jul 12);
- of the fixes, only #24411 (Jun 11) was in.

**Trial A's 0/29 warm hits is therefore not a clean measurement of current llama.cpp.** It measured the
regression window. Current release builds (b10470, 2026-08-17) have #24176 + #24411 + #25472; they still
lack #25592. **[upstream]**

### 3.2 What GGUF would buy if #25592 lands (or is cherry-picked)

llama.cpp's KV cache *is* in place — there is no per-store copy at all; a checkpoint is
`llama_state_seq_save`-style bytes into host RAM (`--cache-ram`), and restore is a memcpy back. So the
memory failure mode we have on MLX simply does not exist there: **6-bit and 8-bit could hold a warm cache
where MLX cannot.** The counter-weight is unchanged from our own data: `REPORT.md` decode at Qwen3.8-27B is
MLX-4bit 27.2 t/s vs GGUF UD-Q4_K_XL+MTP 12.1 / Q8_0 13.5 t/s, and Trial A measured 13.3 t/s decode with
~290 t/s prefill. **Nothing merged since July changes decode speed.** So GGUF remains ~half MLX per token,
and the trade is "half decode speed, but a warm cache that never OOMs" — worth it only if the cache
actually stays warm, which today requires an unmerged PR. **[upstream] + repo data**

---

## 4. Addendum: why 35B A3B *feels* warm on llama.cpp, and what that proves

### (a) Strict-prefix extension works in place for hybrid; only rollback is broken — **[code, upstream source]**

From `tools/server/server-context.cpp` on current master (fetched today), the reuse decision is:

```cpp
llama_pos pos_next = slot.prompt.tokens.pos_next(n_past);
const bool has_new_tokens = (n_past < slot.task->n_tokens());
// the largest pos_min required for a checkpoint to be useful
const auto pos_min_thold = std::max(0, pos_next - n_swa - (has_new_tokens ? 0 : 1));

if (n_past > 0 && n_past <= slot.prompt.n_tokens()) {
    const auto pos_min = llama_memory_seq_pos_min(llama_get_memory(ctx_tgt), slot.id);
    ...
    if (pos_min >= pos_min_thold) {
        // search for a context checkpoint  ... else "forcing full prompt re-processing"
    }
}
```

`n_past` is the common prefix (`slot.prompt.tokens.get_common_prefix(input_tokens)`). For a **strict
extension** — the cached tokens are a prefix of the new prompt, nothing removed — `n_past ==
slot.prompt.n_tokens()`, `has_new_tokens` is true, and for a recurrent memory `pos_min` is the position of
the single stored state, i.e. `n_past - 1`, which is **below** `pos_min_thold ≈ pos_next = n_past`. The
`pos_min >= pos_min_thold` branch is not taken at all: no checkpoint search, no `llama_memory_seq_rm`, no
warning. The slot just keeps decoding forward on the state it already has. **Strict-extension reuse works
for hybrid/recurrent models.**

The moment the prompt *diverges* (`n_past < slot.prompt.n_tokens()`), the server must roll the sequence
back to `n_past`. For recurrent memory `pos_min` is the full sequence length, so the condition fires, a
checkpoint at or below `n_past` is required, and if none is valid you get the exact line in our logs:

> `forcing full prompt re-processing due to lack of cache data (likely due to SWA or hybrid/recurrent
> memory, see .../pull/13194#issuecomment-2868343055)`
> — `results/overnight/llamacpp-server.log`

This reconciles the two local observations perfectly:

- **Bench conditions (strict append) were warm**: `results/warmcache.jsonl` — qwen3.8-27b, `arm: gguf`,
  `quant: q8`, turn 1 TTFT **76,932 ms**, turns 2-5 **1,087 / 1,623 / 2,301 / 998 ms**. Same hybrid arch,
  same engine. `results/overnight/STAGES.md:15-16` records this as *"llama.cpp caching demonstrably works
  on 3.8 hybrids in bench conditions."*
- **Claude Code (mutating prompts) got 0/29**: `STAGES.md:45-47` — *"Warmth: 29 requests, 0 warm hits …
  Verified prompts were clean 91.8% extensions."* 91.8% common prefix is precisely **not** a strict
  extension: 8.2% of the tail differs, a rollback is required, and hybrid memory cannot do it. The later
  MLX diagnostic found the cause is real and systematic — `STAGES.md:66-71`, cold re-prefills every ~5th
  request at near-constant ~1,200-token gaps, i.e. an accumulating block in Claude Code's system prompt
  that `stripVolatile` did not cover.

So: **in-place strict-extension reuse is fine on hybrids; rollback/checkpoint-restore is the broken half**,
and Claude Code's prompts are rollback-shaped by construction. #25592's second half targets exactly this
("agentic clients strip the reasoning of the previous reply so the next request diverges right at the end
of the previous prompt").

### (b) Is Qwen 3.6 35B A3B hybrid, or plain attention MoE? — **hybrid. [code]**

`results/append_only.jsonl` line 2: `"model": "qwen3.6-35b-a3b", "cache_kinds": ["ArraysCache","KVCache"]`
— MLX builds `ArraysCache` for GatedDeltaNet layers and `KVCache` for full-attention layers, so the 35B is
the same hybrid family as the 27B, just MoE (~3B active). llama.cpp agrees: `STAGES.md:17-18` logs the
`hybrid/recurrent memory` line on qwen3.6-35b, and upstream `#24055` has independent reports on
"Qwen3.6-35B-A3B" and "Qwen3.6-35b-a3b-mtp on b9694". There is no contradiction to resolve: it *is*
hybrid.

The warm feel has two independent explanations, both of which apply:
1. **Strict-extension turns genuinely reuse the cache in place** (§4a) — a plain multi-turn chat that only
   appends is warm on hybrids.
2. **When it does cold-reprocess, an MoE reprocesses ~4.5× faster.** Our own measurement, identical prompt
   and harness: cold 23k-token prefill in MLX is **56.6 s on 27B dense vs 12.6 s on 35B-A3B**
   (`results/append_only.jsonl`, `ttft_cold_s`). A "couple of seconds" answer on the 35B is consistent
   with *either* a warm hit *or* a cheap cold reprocess of a moderate context; on the 27B the same cold
   path is a minute, which is why only the 27B feels catastrophic. Distinguishing the two on the 35B needs
   the server log's `prompt eval time / n_tokens` line, not the wall clock.

### (c) Does "35B A3B as the warm daily driver, 27B on demand" beat a 27B full-context path?

On effort/risk, **yes — it is the best value move available today**, and it is folded into the ranking
below as Path 1. Rationale: it needs zero patches; the arch is hybrid either way so it inherits the same
strict-extension warmth; its decode is already the fastest thing we measured (`REPORT.md:22` — 35B A3B
MLX-8bit **70.7 t/s**, quality 8.4/10, versus 27.2 t/s for 27B MLX-4bit); and its cold-prefill penalty is
4.5× smaller, which converts the residual cache misses from fatal to tolerable. The cost is quality on the
hardest tasks (35B A3B MLX-8bit 8.4/10 is *graded above* the GGUF arms but the 27B is the reference for
hard work in this repo) and roughly double the resident weights at 8-bit, which competes with the KV
budget. It does **not** solve the 27B snapshot bug — it routes around it.

---

## 5. Practical mitigations for mlx_vlm 0.6.17 as it stands

**Does the disk tier avoid the GPU transient? No — concretely, it makes the peak worse. [code]**

`store_exact_cache` (`apc.py:3133-3195`) clones on GPU at line 3146 *before* touching the disk branch, and
then hands **the already-cloned GPU arrays** to `self.disk.save_exact_cache(key, token_tuple, extra_hash,
copied)` at line 3186. `DiskBlockStore.save_exact_cache` (`apc.py:2400-2421`) wraps them in a
`_DiskExactCacheSnapshot` and enqueues to a background writer (`_enqueue_exact_snapshot`, `apc.py:2466`),
so those GPU buffers stay alive until the writer drains. The writer then slices them again
(`_snapshot_exact_cache_entry`, `apc.py:2564-2646`: `arrays[f"{prefix}_k"] = c.keys[..., :off, :]`) and
does a **second** `mx.eval(list(arrays.values()))` at `apc.py:2672` before `mx.save_safetensors`.
Setting `APC_EXACT_CACHE_ENTRIES=0` with the disk tier on removes the *retained* RAM copy but **not** either
GPU clone. There is no disk path that streams layer-by-layer off the live cache.

**Is there a chunked `mx.eval` in the clone path? No. [code]**
`_clone_prompt_cache_for_apc` accumulates every layer's tensors and evaluates once (`apc.py:304-322`);
same for `_clone_layer_major_kv_cache_for_apc` (`apc.py:345`). The chunking env knobs that exist —
`APC_DISK_EVAL_BLOCK_CHUNK` (`apc.py:2904`, default 8) and `APC_DISK_LOAD_BLOCK_CHUNK` (`apc.py:2913`) —
apply only to *disk block loads*, not to stores and not to exact entries.

**Is there a smaller checkpoint-guard interval? No — the knob does the opposite of what its name suggests. [code]**
`APC_EXACT_PREFIX_GUARD_TOKENS` (default 16, `apc.py:2880`) is a *tail* guard: the checkpoint is taken at
`len(prompt) - 16` (`ar.py:2363` → `adjust_prefix_to_text_suffix_boundary`, `apc.py:480`). Raising it
shrinks the snapshot by that many tokens and nothing more; there is no periodic-checkpoint mode. The
retention knob is `APC_EXACT_CACHE_ENTRIES` (default 2, `apc.py:2878`) and — as PR #1576 argues — it bounds
*count*, not bytes.

**Existing local patch, for the record.** `local-setup/scripts/apply-latest-only-patch` (per
`STAGES.md:11-13`) gates the two full-prompt stores behind `APC_SKIP_FULL_STORE`. `STAGES.md:52-55`
correctly notes it *"does NOT halve snapshot BYTES; it halves per-cold-prefill big-copy events (2→1)"* —
it removes the `ar.py:2115` harvest store, not the `ar.py:1872` checkpoint store, and not the double clone
inside that one.

---

## 6. Ranked paths to "full-context warm cache on this machine"

Ordered by (value ÷ effort) with risk called out. Effort is wall-clock for one focused session.

### 1. Kill the double clone in the store path — **do this first**
**Effort: ~1 hour. Risk: low. Gain: removes ~2.9 GB from every store's transient (≈ one third of peak).**
Port PR #2072's idea only: add `take_ownership: bool = False` to `store_exact_cache` (`apc.py:3133`), skip
line 3146 when set, and pass `take_ownership=True` from both call sites (`ar.py:1875`, `ar.py:2117`) since
`snapshot_prompt_cache_row` already returns a detached, fully-evaluated clone that `ar.py` drops
immediately. Do **not** port the whole PR — it targets a post-#1960 `main` with an `apc_coordinator.py`
that 0.6.17 does not have. Verify with the PR's own test shape (`monkeypatch` `_clone_prompt_cache_for_apc`
to raise, assert the store still succeeds). Risk is that a future caller mutates the passed cache; the
default stays `False`, so it is opt-in per call site.

### 2. Chunk the `mx.eval` in `_clone_prompt_cache_for_apc`
**Effort: ~1 hour. Risk: low. Gain: unknown but plausibly decisive for the panic, not the OOM.**
Evaluate per cache entry (or per N entries) instead of one 32-tensor command buffer at `apc.py:321`.
Total resident bytes are unchanged, so this does not fix "we need 2× KV"; it fixes "we ask Metal for one
2.9 GB command buffer", which is the shape mlx-lm#1480 blames for long-context Metal aborts and which
PR #2072 already adopts on the *restore* side ("materialize/requantize one restored layer at a time and
release its source"). The 6-bit kernel panic (IOGPUMemory.cpp:550) is a driver-level failure on a huge
allocation, so this is the cheapest shot at it. Pairs naturally with #1; do both in one patch.

### 3. Cap the snapshot by size, not count (port #1576)
**Effort: ~1 hour. Risk: low, but it *gives up* warmth above the cap. Gain: guaranteed no OOM.**
`APC_EXACT_MAX_PROMPT_TOKENS`: in `_store_apc_exact_checkpoints` (`ar.py:1861`), skip the store when
`checkpoint_len` exceeds the cap. This is a safety rail, not a solution — set it just under the measured
cliff (e.g. 38k on 4-bit) so long conversations degrade to cold prefill instead of dying. Worth having
regardless of which other path wins, because every other path has a tail risk and this one bounds it.

### 4. Incremental/delta exact store — hybrid per-layer mode
**Effort: 1-2 days. Risk: medium-high (correctness). Gain: the actual fix — ~19× less store churn.**
Route `KVCache` layers through the existing block store (`store_kv_blocks`, `apc.py:3321`, which is already
chunk-evaluated and disk-backed) and `ArraysCache` layers through a full checkpoint copy, intersecting hits
at the recurrent checkpoint boundary — i.e. re-implement vLLM's `HybridKVCacheCoordinator` +
`--mamba-cache-mode align` (PR #46384) in `apc.py`. Requires relaxing `model_apc_mode`'s all-or-nothing
gate (`apc.py:4168`) to a per-entry decision, a per-layer plan in `apc_lookup_plan` (`apc.py:4179`), and a
commit watermark so a trim below it forces a real copy (§1.2). Land it upstream — no one has proposed it.
Validate against vLLM's known trap (#45238): if the recurrent checkpoint lands in request-unique tokens,
*all* reuse must not silently drop to zero — emit a stat, and check `served_tokens` vs `matched_tokens`
(the failure in mlx-vlm#2048 looks exactly like that in the stats).

### 5. Zero-copy snapshot via a freeze watermark
**Effort: 1 day. Risk: high (silent corruption if the invariant breaks). Gain: eliminates the store copy entirely.**
Have the snapshot hold an *un-evaluated* `c.keys[..., :ckpt, :]` and give `KVCache` a `frozen_below`
watermark; live writes go to `[ckpt:]` only, so the frozen region's contents are stable, and any `trim`
below the watermark triggers a real copy at that moment. Cost: the graph node keeps the old full-capacity
buffer alive (a retention cost roughly equal to what we already pay for the copy), but there is **no
transient spike and no giant `mx.eval`**. This is the highest-leverage idea per line of code and the one
most likely to be wrong — it depends on MLX's `__setitem__`/descriptor aliasing behaving as reasoned in
§1.1, which I have inferred from the code's structure, not verified against MLX internals. Prototype it in
isolation (write into a parent, evaluate a stale slice, assert the slice is unchanged) before touching
`apc.py`.

### 6. Switch the daily driver to Qwen 3.6 35B A3B, keep the 27B on demand
**Effort: ~0 (config). Risk: low, quality-only. Gain: 70.7 t/s decode and a 4.5× cheaper cache miss.**
Per §4c. This is the best *non-engineering* move and it composes with every path above — it does not fix
the 27B store bug, it makes the store bug's blast radius survivable while paths 1-5 are built. Take it now
as the interim default; keep the 27B for hard tasks at whatever context the current OOM cliff allows.

### 7. Re-measure GGUF on a current build, and cherry-pick #25592
**Effort: ~2 hours (rebuild + one 15-turn run). Risk: low. Gain: possibly a warm 6/8-bit cache MLX cannot hold.**
Trial A's verdict is void: b9620 (2026-06-13) sat inside the #22929 regression window and predates #24176,
#25472 (§3.1). A current build (≥ b10470) plus a cherry-pick of the still-open #25592 is the only
configuration in which llama.cpp is expected to hold hybrid checkpoints across a diverging agentic prompt.
Because llama.cpp's KV is in place, **there is no per-store copy at all**, so 6-bit/8-bit warm cache is
memory-feasible there and is not on MLX. The unchanged counter-weight: decode stays ~half MLX
(13.3 t/s measured vs 27.2 t/s MLX-4bit; `REPORT.md` decode table), so this wins only if warmth is worth
2× slower tokens. Track #25592; if it merges, this jumps above path 4.

### Not recommended
- **Disk-tier as an OOM fix** — §5, it adds a queue-retained GPU clone and a second `mx.eval`.
- **`APC_EXACT_CACHE_ENTRIES=0` alone** — removes retention, not either transient clone.
- **`--kv-bits 8`** — already rejected in `STAGES.md:100-104` (streaming tool-calls truncate, no
  `finish_reason`), and mlx-vlm#2093 (2026-08-29) reports `--kv-bits` with `--draft-model` silently
  producing dense caches anyway.
- **Raising `APC_EXACT_PREFIX_GUARD_TOKENS`** — shrinks the snapshot by that many tokens, nothing more.

---

## Appendix: primary sources

llama.cpp — [#19794](https://github.com/ggml-org/llama.cpp/issues/19794) ·
[#21831](https://github.com/ggml-org/llama.cpp/issues/21831) ·
[#22384](https://github.com/ggml-org/llama.cpp/issues/22384) ·
[#24055](https://github.com/ggml-org/llama.cpp/issues/24055) ·
[#22929](https://github.com/ggml-org/llama.cpp/pull/22929) ·
[#24176](https://github.com/ggml-org/llama.cpp/pull/24176) ·
[#24411](https://github.com/ggml-org/llama.cpp/pull/24411) ·
[#24797](https://github.com/ggml-org/llama.cpp/pull/24797) ·
[#25472](https://github.com/ggml-org/llama.cpp/pull/25472) ·
[#25592](https://github.com/ggml-org/llama.cpp/pull/25592) ·
[#13194](https://github.com/ggml-org/llama.cpp/pull/13194) ·
[server-context.cpp](https://github.com/ggml-org/llama.cpp/blob/master/tools/server/server-context.cpp)

mlx-vlm — [#1576](https://github.com/Blaizzy/mlx-vlm/pull/1576) ·
[#1835](https://github.com/Blaizzy/mlx-vlm/pull/1835) ·
[#2048](https://github.com/Blaizzy/mlx-vlm/issues/2048) ·
[#2072](https://github.com/Blaizzy/mlx-vlm/pull/2072) ·
[#999](https://github.com/Blaizzy/mlx-vlm/issues/999)

mlx-lm — [#980](https://github.com/ml-explore/mlx-lm/issues/980) ·
[#1480](https://github.com/ml-explore/mlx-lm/issues/1480)

mlx-engine — [cache_wrapper.py](https://github.com/lmstudio-ai/mlx-engine/blob/main/mlx_engine/cache_wrapper.py) ·
[#327](https://github.com/lmstudio-ai/mlx-engine/issues/327) ·
[#177](https://github.com/lmstudio-ai/mlx-engine/issues/177)

vLLM — [#46384](https://github.com/vllm-project/vllm/pull/46384) ·
[#45238](https://github.com/vllm-project/vllm/issues/45238) ·
[#40696](https://github.com/vllm-project/vllm/issues/40696) ·
[#37898](https://github.com/vllm-project/vllm/pull/37898)
