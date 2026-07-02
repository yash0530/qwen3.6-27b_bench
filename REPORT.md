# Local LLM Benchmarks — Multi-Model Analysis

_Speed sweep: 75 runs · full-length: 15 runs · temp 0.6, seed 42, ctx 16384._

## TL;DR

- **Qwen 3.6 27B (Q5)** — peak **17.7 tok/s** at draft-n=3 (1.37x vs off) (57% accept); quality 8.3/10; ~6795 tok/answer.
- **Qwen 3.6 27B (Q6)** — peak **17.5 tok/s** at draft-n=3 (1.61x vs off) (57% accept); quality 8.5/10; ~6781 tok/answer.
- **Qwen 3.6 27B (Q8)** — peak **17.7 tok/s** at draft-n=2 (1.80x vs off) (68% accept); quality 8.7/10; ~6703 tok/answer.
- **Qwen 3.6 27B (MLX-8bit)** (MLX) — **8.0 tok/s**, quality ungraded, ~6468 tok/answer.

## Charts

![01_decode_tok_s.png](results/charts/01_decode_tok_s.png)

![02_speedup.png](results/charts/02_speedup.png)

![03_acceptance.png](results/charts/03_acceptance.png)

![04_prompt_speed.png](results/charts/04_prompt_speed.png)

![05_ttft.png](results/charts/05_ttft.png)

![06_tokens.png](results/charts/06_tokens.png)

![07_quality.png](results/charts/07_quality.png)

![08_quality_vs_speed.png](results/charts/08_quality_vs_speed.png)

![09_runtime_tok_s.png](results/charts/09_runtime_tok_s.png)

## Speed sweep (mean over questions)

| model/quant | draft-n | tok/s | ±sd | accept % | prompt tok/s | TTFT ms |
|---|---|---|---|---|---|---|
| Qwen 3.6 27B (Q5) | off | 12.9 | 0.2 | - | 257 | 814 |
| Qwen 3.6 27B (Q5) | mtp1 | 14.9 | 0.3 | 80 | 247 | 849 |
| Qwen 3.6 27B (Q5) | mtp2 | 13.4 | 0.6 | 67 | 246 | 853 |
| Qwen 3.6 27B (Q5) | mtp3 | 17.7 | 1.0 | 57 | 242 | 864 |
| Qwen 3.6 27B (Q5) | mtp4 | 17.2 | 1.2 | 49 | 241 | 868 |
| Qwen 3.6 27B (Q6) | off | 10.8 | 0.0 | - | 275 | 760 |
| Qwen 3.6 27B (Q6) | mtp1 | 15.3 | 0.1 | 79 | 266 | 782 |
| Qwen 3.6 27B (Q6) | mtp2 | 14.1 | 0.2 | 68 | 251 | 829 |
| Qwen 3.6 27B (Q6) | mtp3 | 17.5 | 0.6 | 57 | 244 | 856 |
| Qwen 3.6 27B (Q6) | mtp4 | 16.8 | 0.7 | 48 | 246 | 850 |
| Qwen 3.6 27B (Q8) | off | 9.9 | 0.0 | - | 282 | 745 |
| Qwen 3.6 27B (Q8) | mtp1 | 15.4 | 0.4 | 79 | 277 | 753 |
| Qwen 3.6 27B (Q8) | mtp2 | 17.7 | 0.6 | 68 | 275 | 761 |
| Qwen 3.6 27B (Q8) | mtp3 | 17.7 | 0.9 | 57 | 257 | 817 |
| Qwen 3.6 27B (Q8) | mtp4 | 16.9 | 1.2 | 49 | 253 | 829 |

## Full-length output (8192 cap, off config)

| model/quant | total tok | thinking tok | answer tok | answers completed |
|---|---|---|---|---|
| Qwen 3.6 27B (Q5) | 6795 | 4413 | 2378 | 5/5 |
| Qwen 3.6 27B (Q6) | 6781 | 4397 | 2379 | 5/5 |
| Qwen 3.6 27B (Q8) | 6703 | 4349 | 2350 | 5/5 |
| Qwen 3.6 27B (MLX-8bit) | 6468 | 4024 | 2443 | 4/4 |

## Output determinism (MTP correctness probe, #23302)

- 6/60 MTP runs produced a different output than their MTP-off baseline (fixed seed). 0 = MTP is output-preserving on this build; >0 flags the known determinism bug.

## Hosting recommendation

On this machine, serve **Qwen 3.6 27B (Q8)** with peak speed. Launch line:

```bash
llama-server -m /Users/yash/Models/qwen3.6-27b-mtp-q8/Qwen3.6-27B-Q8_0.gguf \
  --spec-type draft-mtp --spec-draft-n-max 2 \
  -c 16384 -ngl 99 -fa on -np 1 --jinja --reasoning-format deepseek \
  --temp 0.6 --top-p 0.95 --top-k 20 --host 127.0.0.1 --port 8080
```
