| Runtime | Quant | Size | shallow off | shallow MTP | agent off | agent MTP | acc | **best agent** | **@depth** | deep | Full | Swept |
|---|---|---:|---:|---:|---:|---:|---:|---:|:--:|---:|:--:|:--:|
| llama.cpp | `q8` | 29.0 GB | 9.85 | 14.77 | 9.10 | 12.35 | 49% | **12.35** | n=2 | · | yes | · |
| MLX | `mlx8` | 29.5 GB | 9.80 | 16.26 | 9.37 | 13.55 | 49% | **13.55** | blk 3 | · | · | · |
| MLX | `mlx6` | 22.8 GB | 12.85 | 21.08 | 11.92 | 16.62 | 49% | **16.62** | blk 3 | · | · | · |
| MLX | `mlx4` | 16.1 GB | 18.06 | 27.17 | 15.93 | 20.00 | 47% | **20.00** | blk 3 | · | · | · |

**Peak so far: 27.17 tok/s** — MLX `mlx4` at block 3, shallow depth, 56% acceptance.

| Auxiliary run | Status | Result |
|---|:--:|---|
| Append-only cache continuation (MLX) | done | 56.64s → 0.47s (119x), output diverges |
| Machine drift control | pending | stage D |
| Quant agreement vs reference | pending | after stage B |
| Quality grading (rubric) | pending | after stage B |

*321 measurement rows recorded for qwen3.8-27b. `·` = not yet run. Decode figures are tok/s, mean of 5 questions; GGUF MTP at n=2, MLX at block 3. `@depth` is the fastest depth **among those measured so far** — the screen only runs off plus one depth, so it is provisional until the stage C sweep fills in the rest.*
