#!/usr/bin/env python3
"""MLX benchmark harness — in-process mlx_vlm with MTP speculative decoding.

Records into the SAME results/results.jsonl (runtime="mlx_vlm") so the MLX configs land
on the same charts as the GGUF ones.

Why this drives mlx_vlm in-process instead of mlx_lm.server over HTTP — each bullet is a
defect in the previous version of this file that the rewrite removes:

  * **Engine-reported numbers.** GenerationResult carries prompt_tokens / prompt_tps /
    generation_tokens / generation_tps / peak_memory. The old harness re-encoded stripped
    output text to count tokens (systematic undercount) and derived prompt speed as
    prompt_n / TTFT, which folds scheduling and first-token decode into a "prefill" number
    and is not comparable to llama.cpp's prompt_per_second.
  * **One reasoning split, not two.** stream_generate yields raw text containing real
    <think>/</think> markers, exactly like llama.cpp's /completion. So split_think applies
    once, to genuine markers. The old harness re-assembled the server's pre-split
    reasoning/content fields into a fake raw string and re-parsed it, which filed all
    reasoning as *answer* whenever generation was truncated mid-thought.
  * **Cache isolation.** A fresh prompt cache per generation, asserted via
    cached_tokens == 0. The GGUF control runs cache_prompt: False; the old MLX runs let
    mlx_lm.server accumulate a 2.67 GB cross-request cache.
  * **Seeded.** The seed is passed through, so MLX runs are reproducible like GGUF ones.
  * **Acceptance is recorded.** Read off the drafter's accept_lens / draft_lens instead of
    being written as None.

MTP comes from mlx_vlm.speculative (--draft-kind mtp), using the separately-published bf16
drafter checkpoints. bf16 matters: quantized MTP heads are reported to collapse acceptance
on MoE models.

Run under the venv:
  .mlxenv/bin/python bench_mlx.py --model qwen3.6-27b
  .mlxenv/bin/python bench_mlx.py --model qwen3.6-27b --smoke
"""
import argparse
import hashlib
import json
import os
import sys
import time
from datetime import datetime, timezone

import config as C
import tiers as T
from questions import QUESTIONS, QUESTION_BY_ID
from bench import log, append_record, split_think, ensure_dirs, consolidate


def mlx_done_keys(path):
    """Completed-run keys, tier- and kv-aware.

    bench.done_keys predates the tier/kv sweep and keys only on
    (phase, pass, model, quant, draft_n, question). That would treat the same question at
    shallow/agent/deep as one cell, so a restarted sweep would either skip work it never
    did or append duplicates that then get averaged into the summary.
    """
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if (r.get("error") is not None or r.get("runtime") != "mlx_vlm"
                    or r.get("smoke")):
                continue
            seen.add((r.get("phase"), r.get("model"), r.get("quant"),
                      r.get("draft_block_size"), r.get("kv_bits"),
                      r.get("prompt_tier"), r.get("question_id")))
    return seen


def _lazy_imports():
    """Imported late so `--help` and arg errors don't pay the MLX import cost."""
    from mlx_vlm import load, stream_generate, apply_chat_template
    from mlx_vlm.speculative.drafters import load_drafter, validate_drafter_compatibility
    return load, stream_generate, apply_chat_template, load_drafter, validate_drafter_compatibility


# ------------------------------------------------------------------ spec counters
def reset_spec_counters(draft_model):
    """Zero the drafter's per-round accounting so each question is measured alone."""
    if draft_model is None:
        return
    for attr in ("accept_lens", "draft_lens"):
        if hasattr(draft_model, attr):
            setattr(draft_model, attr, [])


def read_spec_counters(draft_model):
    """Return (drafted_total, accepted_total, acceptance_rate) or (None, None, None).

    Mirrors mlx_vlm.speculative.common._format_speculative_stats: acceptance is
    accepted drafts / drafted tokens, which is the same quantity llama.cpp reports as
    draft_n_accepted / draft_n — so the two runtimes are directly comparable on chart 03.
    """
    if draft_model is None:
        return None, None, None
    accept_lens = getattr(draft_model, "accept_lens", None) or []
    draft_lens = getattr(draft_model, "draft_lens", None) or []
    if not accept_lens:
        return None, None, None
    accepted = sum(accept_lens)
    drafted = sum(draft_lens) if draft_lens else None
    if not drafted:
        return None, int(accepted), None
    return int(drafted), int(accepted), accepted / drafted


# ------------------------------------------------------------------- one generation
def generate_once(stream_generate, model, processor, prompt, n_predict, draft_model,
                  draft_kind, draft_block_size, kv_bits, reasoning, tier="shallow",
                  sampling=None):
    """Run one generation, returning engine-reported stats plus the raw text."""
    temp, top_p, top_k = sampling or (C.TEMP, C.TOP_P, C.TOP_K)
    kwargs = dict(
        max_tokens=n_predict,
        temperature=temp,
        top_p=top_p,
        top_k=top_k,
        seed=C.SEED,
        enable_thinking=bool(reasoning),
        prefill_step_size=C.PREFILL_STEP_SIZE,
    )
    if kv_bits:
        kwargs.update(kv_bits=kv_bits, kv_group_size=C.KV_GROUP_SIZE,
                      kv_quant_scheme=C.KV_QUANT_SCHEME,
                      quantized_kv_start=C.quantized_kv_start_for(tier))
    if draft_model is not None and draft_block_size:
        kwargs.update(draft_model=draft_model, draft_kind=draft_kind,
                      draft_block_size=draft_block_size)

    reset_spec_counters(draft_model)

    # Seed MLX's global RNG explicitly. Passing `seed=` to stream_generate is NOT enough:
    # ar.py only builds its seeded sampler when top_k == DEFAULT_TOP_K (0), and we pass
    # top_k=20 for parity with llama.cpp, so it silently falls through to make_sampler(),
    # which takes no seed. Seeding globally keeps sampling parity (top_k=20 on both arms)
    # while still making runs reproducible — which the MTP determinism probe requires.
    import mlx.core as mx
    mx.random.seed(C.SEED)

    pieces = []
    t0 = time.time()
    ttft_ms = None
    last = None
    for resp in stream_generate(model, processor, prompt, **kwargs):
        if resp.text:
            if ttft_ms is None:
                ttft_ms = (time.time() - t0) * 1000.0
            pieces.append(resp.text)
        last = resp
    wall_ms = (time.time() - t0) * 1000.0

    drafted, accepted, acc_rate = read_spec_counters(draft_model)
    return {
        "raw_text": "".join(pieces),
        "ttft_ms": ttft_ms,
        "wall_ms": wall_ms,
        "prompt_n": getattr(last, "prompt_tokens", None),
        "prompt_per_second": getattr(last, "prompt_tps", None),
        "predicted_n": getattr(last, "generation_tokens", None),
        "predicted_per_second": getattr(last, "generation_tps", None),
        "peak_memory_gb": getattr(last, "peak_memory", None),
        "cached_tokens": getattr(last, "cached_tokens", None),
        "finish_reason": getattr(last, "finish_reason", None),
        "draft_tokens_total": drafted,
        "draft_tokens_accepted": accepted,
        "acceptance_rate": acc_rate,
    }


def config_label(block_size):
    return "off" if not block_size else f"mtp{block_size}"


# --------------------------------------------------------------------------- runner
def run(model_id, model_cfg, quant, block_sizes, kv_bits_opts, tier_names,
        question_ids, n_predict, phase, resume=True, smoke=False):
    ensure_dirs()
    load, stream_generate, apply_chat_template, load_drafter, validate_compat = _lazy_imports()

    model_path = model_cfg["quants"][quant]
    if not os.path.isdir(model_path):
        log(f"skip {model_id}/{quant}: not present at {model_path}")
        return

    log(f"loading target {model_id}/{quant} from {model_path} ...")
    model, processor = load(model_path)

    tokenizer = getattr(processor, "tokenizer", processor)

    def count_tokens(text):
        return len(tokenizer.encode(text))

    built = T.build(QUESTIONS, count_tokens)

    draft_model, draft_kind = None, None
    draft_path = model_cfg.get("draft_model")
    if draft_path and os.path.isdir(draft_path):
        log(f"loading drafter from {draft_path} ...")
        draft_model, draft_kind = load_drafter(draft_path, kind=model_cfg.get("draft_kind"))
        log(f"  drafter kind resolved to {draft_kind!r}")
        validate_compat(model, draft_model, draft_kind)
        log("  drafter is compatible with the target")
    elif draft_path:
        log(f"  !! drafter not present at {draft_path}; MTP configs will be skipped")

    seen = mlx_done_keys(C.RESULTS_JSONL) if resume else set()
    qs = [QUESTION_BY_ID[q] for q in question_ids]
    sampling = C.sampling_for(model_cfg)

    total = len(tier_names) * len(block_sizes) * len(kv_bits_opts) * len(qs)
    done = 0
    t_start = time.time()
    log(f"START [mlx_vlm/{phase}] {model_id}/{quant}: {total} generations "
        f"(tiers={tier_names} blocks={block_sizes} kv_bits={kv_bits_opts} cap={n_predict})")

    for tier in tier_names:
        for kv_bits in kv_bits_opts:
            for bs in block_sizes:
                if bs and draft_model is None:
                    log(f"skip mtp{bs}: no drafter loaded")
                    done += len(qs)
                    continue
                for q in qs:
                    key = (phase, model_id, quant, bs, kv_bits, tier, q["id"])
                    if resume and key in seen:
                        done += 1
                        log(f"  skip (done) {tier} {config_label(bs)} kv{kv_bits or 'fp16'} {q['id']}")
                        continue

                    user_text = T.user_text(built, q["id"], tier)
                    messages = [{"role": "system", "content": q["system"]},
                                {"role": "user", "content": user_text}]
                    # enable_thinking MUST be passed to the *template*, not only to
                    # stream_generate. Without it the Qwen3.6 template emits a
                    # pre-closed "<think>\n\n</think>\n\n" block and the model answers
                    # with no reasoning at all — while llama.cpp (--jinja) reasons by
                    # default. That silently made the two arms do different work.
                    # Qwen 3.8's template adds reasoning_effort (default "xhigh"). Pass it
                    # explicitly when configured: llama-server's --jinja takes the
                    # template default, so leaving it unset here would only match by
                    # coincidence, and a mismatch changes how much the model thinks —
                    # exactly the class of silent divergence that cost a 10h rerun on 3.6.
                    tmpl_kwargs = {}
                    if model_cfg.get("reasoning", True):
                        tmpl_kwargs["enable_thinking"] = True
                    if model_cfg.get("reasoning_effort"):
                        tmpl_kwargs["reasoning_effort"] = model_cfg["reasoning_effort"]
                    prompt = apply_chat_template(
                        processor, model.config, messages, add_generation_prompt=True,
                        **tmpl_kwargs)

                    rec = {
                        "phase": phase, "pass": 1, "model": model_id, "quant": quant,
                        "draft_n": bs, "config": config_label(bs),
                        "runtime": "mlx_vlm", "backend": "mlx_vlm",
                        "draft_kind": draft_kind if bs else None,
                        "draft_block_size": bs, "kv_bits": kv_bits,
                        "quantized_kv_start": C.quantized_kv_start_for(tier) if kv_bits else None,
                        "kv_arm": ("fp16" if not kv_bits else
                                   ("q8-decode-only" if C.QUANTIZED_KV_DEFER else "q8-full")),
                        "prompt_tier": tier,
                        "question_id": q["id"], "category": q["category"],
                        "seed": C.SEED, "n_predict_cap": n_predict,
                        "model_path": model_path,
                        # Smoke runs use a shorter cap and a single question, so their
                        # numbers are not comparable to a real sweep. Tagged rather than
                        # kept out of the file so a failed gate stays inspectable — the
                        # resume keys and the report both filter on this.
                        "smoke": True if smoke else None,
                        "temp": sampling[0], "top_p": sampling[1], "top_k": sampling[2],
                        "reasoning_effort": model_cfg.get("reasoning_effort"),
                        "prompt_sha256": hashlib.sha256(prompt.encode()).hexdigest(),
                        "draft_model_path": (model_cfg.get("draft_model") if bs else None),
                        "ts": datetime.now(timezone.utc).isoformat(), "error": None,
                    }
                    try:
                        gen = generate_once(stream_generate, model, processor, prompt,
                                            n_predict, draft_model, draft_kind, bs,
                                            kv_bits, model_cfg.get("reasoning", True),
                                            tier=tier, sampling=sampling)

                        # Cache isolation is a claim this harness makes; verify it rather
                        # than trust it. A non-zero hit means a previous question's KV
                        # leaked in and the timing is not a cold-prefill measurement.
                        cached = gen.get("cached_tokens")
                        if cached:
                            rec["cache_leak_tokens"] = cached
                            log(f"  !! cache leak: {cached} cached tokens on {q['id']}")

                        thinking, answer = split_think(gen["raw_text"],
                                                       model_cfg.get("reasoning", True))
                        rec.update({k: gen[k] for k in (
                            "prompt_n", "prompt_per_second", "predicted_n",
                            "predicted_per_second", "draft_tokens_total",
                            "draft_tokens_accepted", "acceptance_rate", "ttft_ms",
                            "wall_ms", "peak_memory_gb", "cached_tokens", "finish_reason")})
                        rec.update({
                            "thinking_tokens": count_tokens(thinking) if thinking else 0,
                            "answer_tokens": count_tokens(answer) if answer else 0,
                            "thinking_text": thinking,
                            "answer_text": answer,
                            "raw_text": gen["raw_text"],
                            "output_sha256": hashlib.sha256(gen["raw_text"].encode()).hexdigest(),
                            "output_len_chars": len(gen["raw_text"]),
                        })
                        ts = gen["predicted_per_second"]
                        acc = gen["acceptance_rate"]
                        log(f"  ok [mlx_vlm] {model_id} {quant} {tier} {config_label(bs)} "
                            f"kv{kv_bits or 'fp16'} {q['id']}: {gen['predicted_n']} tok @ "
                            f"{ts:.1f} tok/s"
                            + (f", acc {acc*100:.0f}%" if acc is not None else "")
                            + (f", prefill {gen['prompt_n']}@{gen['prompt_per_second']:.0f} tok/s"
                               if gen["prompt_per_second"] else ""))
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"
                        log(f"  ERR [mlx_vlm] {model_id} {quant} {tier} {config_label(bs)} "
                            f"{q['id']}: {rec['error']}")
                    append_record(rec)
                    done += 1
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed else 0
                    eta = (total - done) / rate if rate else 0
                    log(f"  progress {done}/{total} ({elapsed/60:.0f} min elapsed, "
                        f"ETA {eta/60:.0f} min)")

    log(f"DONE [mlx_vlm] {model_id}/{quant} in {(time.time()-t_start)/60:.1f} min")
    consolidate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6-27b", choices=list(C.MLX_MODELS_CONFIG.keys()))
    ap.add_argument("--quant", default=None, help="restrict to one quant (default: all present)")
    ap.add_argument("--smoke", action="store_true", help="one question, shallow, off vs mtp")
    ap.add_argument("--phase", choices=["speed", "full"], default=None)
    ap.add_argument("--tiers", default=None,
                    help=f"comma-separated subset of {T.TIER_ORDER}")
    ap.add_argument("--blocks", default=None,
                    help="comma-separated draft block sizes, e.g. 0,1,2,3,4")
    ap.add_argument("--kv-bits", default=None,
                    help="comma-separated; 0 means unquantized. e.g. 0,8")
    ap.add_argument("--no-resume", action="store_true")
    args = ap.parse_args()

    model_cfg = C.MLX_MODELS_CONFIG[args.model]
    quants = [args.quant] if args.quant else model_cfg["quant_order"]
    quants = [q for q in quants if os.path.isdir(model_cfg["quants"][q])]
    if not quants:
        log(f"no MLX models present for {args.model}")
        return

    tier_names = args.tiers.split(",") if args.tiers else list(model_cfg.get("tiers", ["shallow"]))
    blocks = ([int(x) for x in args.blocks.split(",")] if args.blocks
              else model_cfg.get("draft_block_ns", [0]))
    kv_opts = ([int(x) or None for x in args.kv_bits.split(",")] if args.kv_bits
               else model_cfg.get("kv_bits_opts", [None]))

    if args.smoke:
        for q in quants[:1]:
            run(args.model, model_cfg, q, [0, model_cfg.get("draft_block_ns", [0, 2])[-1]],
                [None], ["shallow"], ["q2_coding"], C.SMOKE_N_PREDICT, "speed",
                resume=False, smoke=True)
        return

    allq = [q["id"] for q in QUESTIONS]
    for q in quants:
        if args.phase in (None, "speed"):
            run(args.model, model_cfg, q, blocks, kv_opts, tier_names, allq,
                C.SPEED_N_PREDICT, "speed", resume=not args.no_resume)
        if args.phase in (None, "full"):
            # Canonical answers for judging: MTP OFF, unquantized KV, shallow prompt —
            # matching the GGUF canonical (draft_n=0) so the judge compares like with
            # like. Whether MTP alters output is a separate question, answered by the
            # determinism probe rather than folded into the quality score.
            run(args.model, model_cfg, q, [0], [None], ["shallow"], allq,
                C.FULL_N_PREDICT, "full", resume=not args.no_resume)


if __name__ == "__main__":
    main()
