#!/usr/bin/env python3
"""MLX benchmark harness — mirrors bench.py but drives an MLX OpenAI-compatible server.

Records into the SAME results/results.jsonl (with runtime="mlx") so the MLX quants land
on the same charts as the GGUF quants. MLX (plain mlx_lm/mlx_vlm) has no MTP, so each model
is a single config (draft_n=0). Run under the venv: .mlxenv/bin/python bench_mlx.py ...

Token counts come from the model's own tokenizer (transformers AutoTokenizer); speed/TTFT
are timed from the streamed SSE so the numbers are comparable to the llama.cpp timings.
"""
import argparse
import hashlib
import json
import os
import signal
import socket
import subprocess
import sys
import time
import urllib.request
from datetime import datetime, timezone

import config as C
from questions import QUESTIONS, QUESTION_BY_ID
from bench import log, append_record, split_think, ensure_dirs, consolidate
from transformers import AutoTokenizer

MLX_PORT = 8090
MLX_BASE = f"http://127.0.0.1:{MLX_PORT}"
HEALTH_TIMEOUT_S = 600     # 29 GB MLX model can take a while to load
REQ_TIMEOUT_S = 1800
STREAM_GAP_TIMEOUT_S = 240

SMOKE_N_PREDICT = 512


def server_argv(backend, model_path, port):
    if backend == "mlx_vlm":
        return [sys.executable, "-m", "mlx_vlm", "server", "--model", model_path, "--port", str(port)]
    return [sys.executable, "-m", "mlx_lm", "server", "--model", model_path, "--port", str(port),
            "--host", "127.0.0.1"]


class MlxServer:
    def __init__(self, model_id, quant, backend, model_path):
        self.model_id = model_id
        self.quant = quant
        self.backend = backend
        self.model = model_path
        self.log_path = os.path.join(C.LOGS, f"mlxserver_{model_id}_{quant}.log")
        self.proc = None

    def start(self):
        argv = server_argv(self.backend, self.model, MLX_PORT)
        self.logf = open(self.log_path, "ab")
        self.logf.write(f"\n== launch {datetime.now().isoformat()} ==\n{' '.join(argv)}\n".encode())
        self.logf.flush()
        self.proc = subprocess.Popen(argv, stdout=self.logf, stderr=self.logf)
        deadline = time.time() + HEALTH_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                log(f"  !! mlx server exited early (code {self.proc.returncode}); see {self.log_path}")
                return False
            try:
                with urllib.request.urlopen(MLX_BASE + "/v1/models", timeout=4) as r:
                    if r.status == 200:
                        return True
            except Exception:
                pass
            time.sleep(2.0)
        log("  !! mlx server health timeout")
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait(timeout=10)
        try:
            self.logf.close()
        except Exception:
            pass
        deadline = time.time() + 60
        while time.time() < deadline:
            s = socket.socket(); s.settimeout(0.5)
            free = s.connect_ex(("127.0.0.1", MLX_PORT)) != 0
            s.close()
            if free:
                return
            time.sleep(0.5)


def stream_chat(messages, n_predict, model_id):
    payload = {
        "messages": messages, "max_tokens": n_predict,
        "temperature": C.TEMP, "top_p": C.TOP_P, "top_k": C.TOP_K,
        "stream": True, "model": model_id,
    }
    req = urllib.request.Request(MLX_BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; reasoning = []; content = []
    with urllib.request.urlopen(req, timeout=STREAM_GAP_TIMEOUT_S) as resp:
        for raw in resp:
            line = raw.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            body = line[5:].strip()
            if body == "[DONE]":
                break
            try:
                chunk = json.loads(body)
            except Exception:
                continue
            delta = (chunk.get("choices") or [{}])[0].get("delta", {})
            r = delta.get("reasoning") or delta.get("reasoning_content") or ""
            c = delta.get("content") or ""
            if r or c:
                if t_first is None:
                    t_first = time.time()
                if r:
                    reasoning.append(r)
                if c:
                    content.append(c)
    t_end = time.time()
    return {"reasoning": "".join(reasoning), "content": "".join(content),
            "t0": t0, "t_first": t_first, "t_end": t_end}


def run(model_id, model_cfg, quants, question_ids, n_predict, phase, backend, tokenizers):
    ensure_dirs()
    qs = [QUESTION_BY_ID[q] for q in question_ids]
    for quant in quants:
        model_path = model_cfg["quants"][quant]
        if not os.path.isdir(model_path):
            log(f"skip {model_id}/{quant}: model not present at {model_path}")
            continue
        tok = tokenizers[quant]
        srv = MlxServer(model_id, quant, backend, model_path)
        log(f"launch [mlx/{phase}] {model_id} {quant} ({backend}) ...")
        if not srv.start():
            srv.stop(); continue
        try:
            stream_chat([{"role": "user", "content": "hi"}], 16, model_path)
            log(f"  warmed up {model_id}/{quant}")
        except Exception as e:
            log(f"  warmup failed (continuing): {e}")
        for q in qs:
            messages = [{"role": "system", "content": q["system"]},
                        {"role": "user", "content": q["user"]}]
            rec = {"phase": phase, "pass": 1, "model": model_id, "quant": quant, "draft_n": 0,
                   "config": "mlx", "runtime": "mlx", "backend": backend,
                   "question_id": q["id"], "category": q["category"],
                   "seed": C.SEED, "n_predict_cap": n_predict,
                   "model_path": model_path,
                   "ts": datetime.now(timezone.utc).isoformat(), "error": None}
            try:
                gen = stream_chat(messages, n_predict, model_path)
                thinking = gen["reasoning"].strip()
                answer = gen["content"].strip()
                raw = thinking + (("\n</think>\n\n" + answer) if answer else "")
                
                # Use split_think helper if needed, but since MLX streams them split,
                # we just format it as raw first, then run split_think to be sure
                thinking, answer = split_think(raw, model_cfg["reasoning"])
                
                think_tok = len(tok.encode(thinking)) if thinking else 0
                ans_tok = len(tok.encode(answer)) if answer else 0
                comp_tok = think_tok + ans_tok
                ttft_ms = (gen["t_first"] - gen["t0"]) * 1000 if gen["t_first"] else None
                dec_s = (gen["t_end"] - gen["t_first"]) if gen["t_first"] else None
                tok_s = comp_tok / dec_s if dec_s and dec_s > 0.05 else None
                try:
                    rendered = tok.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
                    prompt_n = len(tok.encode(rendered))
                except Exception:
                    prompt_n = None
                prompt_s = (prompt_n / (ttft_ms / 1000)) if (prompt_n and ttft_ms) else None
                rec.update({
                    "predicted_n": comp_tok, "predicted_per_second": tok_s,
                    "prompt_n": prompt_n, "prompt_per_second": prompt_s,
                    "draft_tokens_total": None, "draft_tokens_accepted": None,
                    "acceptance_rate": None, "ttft_ms": ttft_ms,
                    "wall_ms": (gen["t_end"] - gen["t0"]) * 1000,
                    "thinking_tokens": think_tok, "answer_tokens": ans_tok,
                    "thinking_text": thinking, "answer_text": answer, "raw_text": raw,
                    "output_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                    "output_len_chars": len(raw),
                })
                ts = f"{tok_s:.1f}" if tok_s else "?"
                log(f"  ok [mlx] {model_id} {quant} {phase} {q['id']}: {comp_tok} tok @ {ts} tok/s"
                    + (f", ttft {ttft_ms:.0f}ms" if ttft_ms else ""))
            except Exception as e:
                rec["error"] = f"{type(e).__name__}: {e}"
                log(f"  ERR [mlx] {model_id} {quant} {phase} {q['id']}: {rec['error']}")
            append_record(rec)
        srv.stop()
    consolidate()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.6-27b", choices=list(C.MLX_MODELS_CONFIG.keys()),
                    help="Model to benchmark (default: qwen3.6-27b)")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--quant", default=None)
    ap.add_argument("--backend", choices=["mlx_lm", "mlx_vlm"], default="mlx_lm")
    ap.add_argument("--phase", choices=["speed", "full"], default=None)
    args = ap.parse_args()

    model_id = args.model
    model_cfg = C.MLX_MODELS_CONFIG[model_id]

    quants = [args.quant] if args.quant else model_cfg["quant_order"]
    quants = [q for q in quants if os.path.isdir(model_cfg["quants"][q])]
    if not quants:
        log(f"no MLX models present for {model_id} yet"); return
    tokenizers = {q: AutoTokenizer.from_pretrained(model_cfg["quants"][q]) for q in quants}

    if args.smoke:
        run(model_id, model_cfg, quants[:1], ["q2_coding"], SMOKE_N_PREDICT, "speed", args.backend, tokenizers)
        return
    allq = [q["id"] for q in QUESTIONS]
    if args.phase in (None, "speed"):
        run(model_id, model_cfg, quants, allq, C.SPEED_N_PREDICT, "speed", args.backend, tokenizers)
    if args.phase in (None, "full"):
        run(model_id, model_cfg, quants, allq, C.FULL_N_PREDICT, "full", args.backend, tokenizers)


if __name__ == "__main__":
    main()
