# Local LLM Benchmark Suite

Benchmark orchestrator for open-weight LLMs on Apple Silicon, comparing **llama.cpp (GGUF)**
against **MLX (`mlx_vlm`)** on the same models, the same prompts, and the same bytes.

Measures decode throughput, prefill throughput, TTFT, speculative-decoding (MTP) acceptance,
reasoning vs. answer token volume, **prompt-depth scaling**, **warm-cache TTFT**,
**concurrency**, and answer quality via a pluggable judge.

---

## Headline result (Qwen 3.6 27B & 35B A3B, 8-bit, M5 Pro 64 GB)

Measured over the full matrix; see `REPORT.md` for the complete tables.

| Model | Verdict today | With an append-only cache layer |
|---|---|---|
| **35B A3B** | llama.cpp | **MLX** — 17.5 s vs 19.1 s per warm 1024-token turn |
| **27B dense** | **llama.cpp** | llama.cpp — 66.6 s vs 76.9 s |

The decider is **prompt-prefix reuse**, not raw speed. MLX wins cold prefill at every depth
and wins decode on the MoE, but `mlx_vlm.server` re-renders the chat template each request,
forcing a cache rewind that Qwen 3.6's hybrid cache cannot do
([ml-explore/mlx-lm#980](https://github.com/ml-explore/mlx-lm/issues/980)) — so it re-prefills
the whole preamble every turn. Feeding the cache append-only instead makes MLX **64–115x**
faster warm and, on the 35B, byte-identical to a cold run.

---

## Why the comparison is trustworthy

Cross-engine benchmarks fail silently far more often than they fail loudly. Four safeguards
exist because each one caught a real error here:

- **`validate_parity.py`** — gates any GGUF-vs-MLX claim. Checks reasoning-engagement parity
  between arms, prompt-length parity per depth tier, non-empty answers, acceptance coverage,
  cache isolation, and degenerate configs. It retroactively flags an earlier run at
  **95% vs 0% reasoning**, when `enable_thinking` was reaching `stream_generate` but not the
  chat template and MLX was silently answering without reasoning.
- **`drift_check.py`** — re-measures an unchanged GGUF cell and reports machine drift. Over
  this run: **−4.5% decode / −5.7% prefill**. Cross-engine margins under ~5% are unresolved.
- **`tiers.py`** — builds each depth tier once and caches it to `results/prompt_tiers.json`,
  so both engines tokenize **identical bytes** rather than prompts that differ by a few tokens.
- **Cache-liveness assertions** — `bench_warmcache.py` queries `/v1/cache/stats` and aborts
  rather than reporting an unverified comparison.

---

## Gotchas found the hard way

`mlx_vlm` ignores misapplied settings silently rather than erroring. Each of these produced
plausible numbers and a wrong conclusion:

| Gotcha | Symptom |
|---|---|
| `enable_thinking` must go to `apply_chat_template`, not just `stream_generate` | Template emits a pre-closed `<think></think>`; model never reasons |
| `seed` is ignored unless `top_k == 0` (`ar.py:291`) | Runs non-reproducible; determinism probe reports false failures |
| `APC_ENABLED` defaults to `"0"` (`apc.py:3769`) | Prompt caching silently off |
| `--draft-block-size 1` drafts `block_size - 1 = 0` tokens | Emits one token and stops; reports an absurd tok/s |
| Quantized KV + MTP at depth | `AttributeError: 'list' object has no attribute 'shape'` (`qwen3_5/language.py:1481`) |
| Raising `prefill_step_size` above the prompt length | Metal OOM at 23k tokens |

Also on the llama.cpp side: `split_think` filed truncated reasoning as *answer*, corrupting
75 of 210 speed rows before it was fixed.

---

## Layout

| File | Purpose |
|---|---|
| `bench.py` | GGUF arm (llama-server), depth tiers, KV-quant sweep |
| `bench_mlx.py` | MLX arm, in-process `mlx_vlm`, MTP via bf16 drafters |
| `bench_concurrency.py` | 1→10 concurrent clients; three arms (mlx / gguf-mtp / gguf-batch) |
| `bench_warmcache.py` | Multi-turn session TTFT — cold vs warm |
| `drift_check.py` | Machine-drift control |
| `validate_parity.py` | Cross-arm parity gate |
| `tiers.py` | Depth-tier prompt builder (shallow ~200 / agent ~23k / deep ~64k) |
| `report.py` | Charts, `summary.json`, `REPORT.md` / `REPORT.html` |
| `judge_export.py` / `apply_scores.py` | Pluggable quality judging per `RUBRIC.md` |
| `run_exhaustive.sh` | Serialized full matrix (~10–12h) |

---

## Running it

```bash
# 0. Fetch MLX targets + bf16 MTP drafters (~68 GB)
.mlxenv/bin/python download_mlx_models.py

# 1. Smoke test both arms
python3            bench.py     --smoke --model qwen3.6-27b
.mlxenv/bin/python bench_mlx.py --smoke --model qwen3.6-27b

# 2. Full matrix, serialized (never run two engines at once — you will measure contention)
./run_exhaustive.sh

# 3. Gate, then report
python3 validate_parity.py && python3 report.py
```

Use the **bf16** MTP drafters. Quantized MTP heads are reported to collapse acceptance on MoE
models; at bf16 the 35B holds ~80%.

---

## Output (`results/`)

- `results.jsonl` / `results.json` — per-run records (timings, acceptance, full texts)
- `prompt_tiers.json` — the shared byte-identical prompts
- `warmcache.jsonl`, `concurrency.jsonl`, `drift.jsonl` — the auxiliary measurements
- `summary.json`, `charts/`, `REPORT.md`, `REPORT.html`
