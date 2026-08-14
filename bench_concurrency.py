#!/usr/bin/env python3
"""Concurrency sweep: aggregate throughput and per-request latency vs. concurrent clients.

This measures a different thing from bench.py / bench_mlx.py, which time one request at a
time. Here N clients hit a live server simultaneously and we record both what the *system*
delivers (aggregate tok/s) and what each *user* experiences (TTFT, end-to-end latency).
Those diverge sharply under load, and only the second one is what you feel at a keyboard.

Why this is the interesting axis
-------------------------------
The two engines genuinely differ here, and LOCAL_LLM_HARNESS.md §8 currently states the
trade as universal:

    "The trade is concurrency *or* MTP — on either engine."

That holds for llama.cpp, which requires `-np 1` for MTP speculative decoding. It does
not hold for mlx_vlm, which does continuous batching *with* speculation
(server/generation.py:853, plus a dedicated speculative coalesce window and batched
MTP walks in speculative/mtp.py). So llama.cpp needs two arms to be represented fairly:

  gguf-mtp    -np 1 + MTP. Requests queue. This is what llm-serve runs today.
  gguf-batch  -np N, MTP off. llama.cpp's best genuinely-concurrent configuration.
  mlx         mlx_vlm.server with MTP. Batching and speculation at the same time.

Reporting only one llama.cpp arm would misrepresent it: arm (a) understates its
throughput, arm (b) understates its single-user latency.

KV precision note: the MLX arm runs fp16 KV. Quantized KV + MTP is broken in mlx_vlm
0.6.3 (see config.py), so a q8 MLX arm cannot use speculation at all.

Usage:
  .mlxenv/bin/python bench_concurrency.py --arm mlx        --model qwen3.8-27b
  python3            bench_concurrency.py --arm gguf-mtp   --model qwen3.8-27b
  python3            bench_concurrency.py --arm gguf-batch --model qwen3.8-27b
"""
import argparse
import json
import os
import signal
import socket
import statistics as st
import subprocess
import sys
import threading
import time
import urllib.request
from datetime import datetime, timezone

import config as C
import tiers as T
from questions import QUESTIONS, QUESTION_BY_ID

CONCURRENCIES = [1, 2, 3, 4, 6, 8, 10]
PORT = 8091
BASE = f"http://127.0.0.1:{PORT}"
RESULTS_JSONL = os.path.join(C.RESULTS, "concurrency.jsonl")

HEALTH_TIMEOUT_S = 900
REQ_TIMEOUT_S = 3600


def log(msg):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)


# --------------------------------------------------------------------------- servers
def server_argv(arm, model_cfg, mlx_cfg, draft_n, np_slots):
    if arm == "mlx":
        # --enable-thinking is required for parity with llama.cpp, which reasons by
        # default under --jinja. Without it the Qwen3.6 template emits a pre-closed
        # <think></think> block and the model skips reasoning entirely.
        argv = [sys.executable, "-m", "mlx_vlm.server",
                "--model", mlx_cfg["quants"]["mlx8"],
                "--host", "127.0.0.1", "--port", str(PORT),
                "--enable-thinking",
                "--prefill-step-size", str(C.PREFILL_STEP_SIZE)]
        if draft_n:
            argv += ["--draft-model", mlx_cfg["draft_model"],
                     "--draft-kind", "mtp", "--draft-block-size", str(draft_n)]
        return argv

    argv = [C.LLAMA_SERVER, "-m", model_cfg["quants"]["q8"],
            "-c", str(C.ctx_for_tier("shallow", 1024) * 4),
            "-ngl", "99", "-fa", "on", "--jinja", "-s", str(C.SEED),
            "--host", "127.0.0.1", "--port", str(PORT), "--metrics", "--no-webui",
            "-np", str(np_slots)]
    if model_cfg.get("reasoning_format", "none") != "none":
        argv += ["--reasoning-format", model_cfg["reasoning_format"]]
    if arm == "gguf-mtp":
        argv += ["--spec-type", "draft-mtp", "--spec-draft-n-max", str(draft_n)]
    else:
        argv += ["--spec-type", "none"]
    return argv


class Server:
    def __init__(self, argv, log_path):
        self.argv = argv
        self.log_path = log_path
        self.proc = None

    def _wait_port_free(self, timeout=120):
        """Block until the port is bindable.

        The previous server's stop() can return while the socket is still in TIME_WAIT
        or the process is still tearing down. Launching immediately then fails with
        "couldn't bind HTTP server socket" and the whole arm silently produces no rows —
        which is exactly how the GGUF warm-cache arm was lost.
        """
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = socket.socket(); s.settimeout(0.5)
            free = s.connect_ex(("127.0.0.1", PORT)) != 0
            s.close()
            if free:
                return True
            time.sleep(1.0)
        log(f"  !! port {PORT} still bound after {timeout}s")
        return False

    def start(self):
        self._wait_port_free()
        self.logf = open(self.log_path, "ab")
        self.logf.write(f"\n== {datetime.now().isoformat()} ==\n{' '.join(self.argv)}\n".encode())
        self.logf.flush()
        self.proc = subprocess.Popen(self.argv, stdout=self.logf, stderr=self.logf)
        deadline = time.time() + HEALTH_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                log(f"  !! server exited early (code {self.proc.returncode}); see {self.log_path}")
                return False
            for path in ("/health", "/v1/models"):
                try:
                    with urllib.request.urlopen(BASE + path, timeout=3) as r:
                        if r.status == 200:
                            return True
                except Exception:
                    pass
            time.sleep(2.0)
        log("  !! health timeout")
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=30)
            except subprocess.TimeoutExpired:
                self.proc.kill(); self.proc.wait(timeout=10)
        try:
            self.logf.close()
        except Exception:
            pass
        deadline = time.time() + 90
        while time.time() < deadline:
            s = socket.socket(); s.settimeout(0.5)
            free = s.connect_ex(("127.0.0.1", PORT)) != 0
            s.close()
            if free:
                return
            time.sleep(0.5)


# ------------------------------------------------------------------------- one client
def one_request(messages, n_predict, model_name, out, idx):
    """Stream one chat completion; record TTFT, token count and wall time."""
    payload = {"messages": messages, "max_tokens": n_predict, "stream": True,
               "model": model_name, "temperature": C.TEMP, "top_p": C.TOP_P}
    req = urllib.request.Request(BASE + "/v1/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    t0 = time.time(); t_first = None; n_chunks = 0; text = []
    err = None
    try:
        with urllib.request.urlopen(req, timeout=REQ_TIMEOUT_S) as resp:
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
                d = (chunk.get("choices") or [{}])[0].get("delta", {})
                piece = (d.get("content") or "") + (d.get("reasoning") or "") \
                        + (d.get("reasoning_content") or "")
                if piece:
                    if t_first is None:
                        t_first = time.time()
                    n_chunks += 1
                    text.append(piece)
    except Exception as e:
        err = f"{type(e).__name__}: {e}"
    t_end = time.time()
    out[idx] = {
        "ttft_ms": (t_first - t0) * 1000 if t_first else None,
        "wall_ms": (t_end - t0) * 1000,
        "decode_ms": (t_end - t_first) * 1000 if t_first else None,
        "chunks": n_chunks,
        "chars": len("".join(text)),
        "error": err,
    }


def run_level(arm, model_id, n_clients, n_predict, tier, built, model_name, draft_n):
    """Fire n_clients concurrent requests and summarise system + per-user behaviour."""
    qs = [QUESTIONS[i % len(QUESTIONS)] for i in range(n_clients)]
    msgs = [[{"role": "system", "content": q["system"]},
             {"role": "user", "content": T.user_text(built, q["id"], tier)}] for q in qs]

    out = [None] * n_clients
    threads = [threading.Thread(target=one_request,
                               args=(msgs[i], n_predict, model_name, out, i))
               for i in range(n_clients)]
    t0 = time.time()
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    wall = time.time() - t0

    good = [r for r in out if r and not r["error"]]
    errs = [r for r in out if r and r["error"]]
    total_chars = sum(r["chars"] for r in good)
    # Index p95 against the TTFT list itself, not len(good): a client that produced no
    # token has ttft None and is filtered out, which would otherwise overrun the list.
    ttfts = sorted(r["ttft_ms"] for r in good if r["ttft_ms"] is not None)
    rec = {
        "ts": datetime.now(timezone.utc).isoformat(),
        "arm": arm, "model": model_id, "tier": tier, "draft_n": draft_n,
        "concurrency": n_clients, "n_predict_cap": n_predict,
        "n_ok": len(good), "n_err": len(errs),
        "batch_wall_s": wall,
        # System view: characters delivered per second across all streams.
        "aggregate_chars_per_s": total_chars / wall if wall else None,
        # User view: what a single client waited for.
        "ttft_ms_mean": st.mean(ttfts) if ttfts else None,
        "ttft_ms_p95": ttfts[int(0.95 * (len(ttfts) - 1))] if ttfts else None,
        "latency_ms_mean": st.mean([r["wall_ms"] for r in good]) if good else None,
        "latency_ms_max": max([r["wall_ms"] for r in good]) if good else None,
        "per_request": out,
        "error_sample": errs[0]["error"] if errs else None,
    }
    with open(RESULTS_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")
    agg = rec["aggregate_chars_per_s"] or 0
    log(f"  c={n_clients:<3} ok={len(good)}/{n_clients} wall={wall:6.1f}s "
        f"agg={agg:8.0f} ch/s  ttft_mean={(rec['ttft_ms_mean'] or 0):7.0f}ms "
        f"lat_max={(rec['latency_ms_max'] or 0):8.0f}ms"
        + (f"  ERR {rec['error_sample'][:40]}" if errs else ""))
    return rec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", required=True, choices=["mlx", "gguf-mtp", "gguf-batch"])
    ap.add_argument("--model", default="qwen3.8-27b", choices=list(C.MODELS_CONFIG.keys()))
    ap.add_argument("--tier", default="shallow", choices=T.TIER_ORDER)
    ap.add_argument("--draft-n", type=int, default=None,
                    help="MTP depth; default = best from the main sweep, per arm")
    ap.add_argument("--n-predict", type=int, default=512)
    ap.add_argument("--levels", default=None, help="comma-separated, default 1,2,3,4,6,8,10")
    args = ap.parse_args()

    os.makedirs(C.RESULTS, exist_ok=True)
    levels = [int(x) for x in args.levels.split(",")] if args.levels else CONCURRENCIES
    model_cfg = C.MODELS_CONFIG[args.model]
    mlx_cfg = C.MLX_MODELS_CONFIG.get(args.model, {})
    built = T.load_cached()

    # llama.cpp cannot do MTP with more than one slot, so gguf-batch has no draft depth.
    draft_n = args.draft_n
    if args.arm == "gguf-batch":
        draft_n = 0
    elif draft_n is None:
        draft_n = 3 if args.arm == "mlx" else 2

    model_name = (mlx_cfg.get("quants", {}).get("mlx8") if args.arm == "mlx"
                  else "qwen-local")

    log(f"=== concurrency sweep: arm={args.arm} model={args.model} tier={args.tier} "
        f"draft_n={draft_n} cap={args.n_predict} levels={levels} ===")

    for n in levels:
        # gguf-batch needs one slot per client, so the server is relaunched per level.
        # The other two arms keep a single server across levels.
        np_slots = n if args.arm == "gguf-batch" else 1
        argv = server_argv(args.arm, model_cfg, mlx_cfg, draft_n, np_slots)
        srv = Server(argv, os.path.join(C.LOGS, f"conc_{args.arm}_{args.model}_c{n}.log"))
        log(f"launch {args.arm} c={n} (-np {np_slots}) ...")
        if not srv.start():
            log(f"  !! skip c={n}: server failed to start")
            srv.stop()
            continue
        try:
            one_request([{"role": "user", "content": "hi"}], 8, model_name, [None], 0)
        except Exception:
            pass
        try:
            run_level(args.arm, args.model, n, args.n_predict, args.tier, built,
                      model_name, draft_n)
        finally:
            srv.stop()

    log("=== concurrency sweep done ===")


if __name__ == "__main__":
    main()
