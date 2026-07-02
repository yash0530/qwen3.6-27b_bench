# Local LLM Benchmark Suite

A generic, extensible local benchmark orchestrator for evaluating open-weights LLMs on Apple Silicon (optimized for macOS, built on `llama.cpp` and `mlx`). 

Measures decode throughput (tok/s), prompt-processing speed, time-to-first-token (TTFT), speculative decoding (MTP) draft acceptance rates, and reasoning (thinking) vs. answer token volumes across realistic, high-depth prompts. Answer quality is evaluated using a pluggable judge (e.g., Claude/Opus) based on a structured rubric.

---

## Supported Models

Models are configured in `config.py`. Out of the box, the suite supports:

1. **Qwen 3.6 27B** (Reasoning, deep knowledge, MTP support)
2. **Gemma 4 31B** (Dense, low-hallucination hedge, standard GGUF)

---

## Benchmark Design

Evaluations are performed in a structured **two-phase** run:
- **Speed Phase**: Evaluates performance over the full sweep grid (quants × speculative draft depths × prompts) using a short `1024` token cap. Since throughput (tok/s) is prefix-independent, this captures speed, TTFT, and draft acceptance rates accurately and efficiently.
- **Full Phase**: Evaluates completions using the full `12288` token cap at the baseline (`draft_n = 0`) configuration to yield full answers for quality grading and token volume counts.

---

## Getting Started

### 1. Requirements
- **llama.cpp** built with Metal support (`llama-server` executable in your PATH or configured in `config.py`).
- **GGUF Models** placed in your `~/Models` directory as mapped in `config.py`. For example:
  - Qwen 3.6 27B GGUFs under `~/Models/qwen3.6-27b-mtp-q8/`
  - Gemma 4 31B GGUFs under `~/Models/gemma4-31b/`

### 2. Run Benchmarks
You can run the benchmarks using the `bench.py` driver:

```bash
# Smoke test (tiny validation run ~3 mins)
python3 bench.py --smoke --model qwen3.6-27b
python3 bench.py --smoke --model gemma4-31b

# Full benchmark run (speed sweep + full phase)
python3 bench.py --model qwen3.6-27b
python3 bench.py --model gemma4-31b
```

### 3. Generate Report
Generate aggregate charts, `summary.json`, and the markdown report (`REPORT.md`):

```bash
python3 report.py
```

### 4. Pluggable Judging (Optional)
Evaluate answer quality using an external LLM judge (configured per `RUBRIC.md`):

```bash
# Export canonical answers needing evaluation
python3 judge_export.py  # Outputs results/judging/to_grade.json

# ... Grader reviews and writes results/judging/scores.json ...

# Merge scores back and regenerate report
python3 apply_scores.py
python3 report.py
```

---

## Output Architecture (`results/`)

- `results.jsonl` / `results.json`: Raw per-run records (full answers, thinking texts, timings).
- `summary.json`: Multi-model aggregates (mean speed, prompt speed, TTFT, acceptance rate).
- `charts/`: Generated performance graphs (decode tok/s, TTFT latency, speedup curves, prompt speeds).
- `REPORT.md`: A human-readable Markdown analysis report summarizing recommendations and aggregates.
