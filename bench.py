#!/usr/bin/env python3
"""Benchmark orchestrator for local LLMs (Qwen, Gemma, etc.).

For every (pass, quant, draft_n) it launches a dedicated llama-server, runs the 5
questions, and records rich per-run metrics to results/results.jsonl (crash-safe,
resumable). Pure stdlib — no third-party deps.

Usage:
  python3 bench.py             # full run for default model (Qwen 3.8 27B)
  python3 bench.py --model gemma4-31b
  python3 bench.py --smoke     # tiny end-to-end validation (~3-5 min)
  python3 bench.py --quant q8  # restrict to one quant
"""
import argparse
import hashlib
import json
import os
import re
import signal
import socket
import subprocess
import sys
import time
import urllib.error
import urllib.request
from datetime import datetime, timezone

import config as C
import tiers as T
from questions import QUESTIONS, QUESTION_BY_ID


# ----------------------------------------------------------------------------- IO
def log(msg: str):
    line = f"[{datetime.now().strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True)
    os.makedirs(C.RESULTS, exist_ok=True)
    with open(C.PROGRESS_LOG, "a") as f:
        f.write(line + "\n")


def ensure_dirs():
    for d in (C.RESULTS, C.LOGS, C.CHARTS, C.JUDGING):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- HTTP
def _post_json(path: str, payload: dict, timeout: float):
    data = json.dumps(payload).encode()
    req = urllib.request.Request(C.BASE_URL + path, data=data,
                                 headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return json.loads(r.read().decode())


def _get(path: str, timeout: float = 5.0):
    with urllib.request.urlopen(C.BASE_URL + path, timeout=timeout) as r:
        return r.read().decode()


def apply_template(messages: list) -> str:
    resp = _post_json("/apply-template", {"messages": messages}, timeout=30)
    return resp.get("prompt", resp.get("content", ""))


def tokenize_count(text: str) -> int:
    if not text:
        return 0
    try:
        resp = _post_json("/tokenize", {"content": text}, timeout=60)
        toks = resp.get("tokens", [])
        return len(toks)
    except Exception:
        return -1


def parse_metrics(text: str) -> dict:
    out = {}
    for ln in text.splitlines():
        if ln.startswith("#") or not ln.strip():
            continue
        parts = ln.split()
        if len(parts) >= 2:
            try:
                out[parts[0]] = float(parts[1])
            except ValueError:
                pass
    return out


# ------------------------------------------------------------------- server ctl
class Server:
    def __init__(self, model_id: str, quant: str, draft_n: int, model_path: str, reasoning_format: str,
                 ctx: int = None, kv_quant: str = None, tier: str = "shallow",
                 draft_sidecar: str = None):
        self.model_id = model_id
        self.quant = quant
        self.draft_n = draft_n
        self.model = model_path
        self.reasoning_format = reasoning_format
        self.ctx = ctx
        self.kv_quant = kv_quant
        self.tier = tier
        self.draft_sidecar = draft_sidecar
        suffix = f"_{tier}" + (f"_kv{kv_quant}" if kv_quant else "")
        self.log_path = os.path.join(C.LOGS, f"server_{model_id}_{quant}_n{draft_n}{suffix}.log")
        self.proc = None

    def start(self):
        cmd = C.server_cmd(self.model, self.draft_n, self.log_path, self.reasoning_format,
                           ctx=self.ctx, kv_quant=self.kv_quant,
                           draft_sidecar=self.draft_sidecar)
        self.logf = open(self.log_path, "ab")
        self.logf.write(f"\n===== launch {datetime.now().isoformat()} =====\n".encode())
        self.logf.write((" ".join(cmd) + "\n").encode())
        self.logf.flush()
        self.proc = subprocess.Popen(cmd, stdout=self.logf, stderr=self.logf)
        return self._wait_healthy()

    def _wait_healthy(self) -> bool:
        deadline = time.time() + C.HEALTH_TIMEOUT_S
        while time.time() < deadline:
            if self.proc.poll() is not None:
                log(f"  !! server exited early (code {self.proc.returncode}); see {self.log_path}")
                return False
            try:
                body = _get("/health", timeout=3)
                if '"ok"' in body or body.strip() == "{}":
                    return True
            except Exception:
                pass
            time.sleep(1.0)
        log(f"  !! server health timeout after {C.HEALTH_TIMEOUT_S}s")
        return False

    def stop(self):
        if self.proc and self.proc.poll() is None:
            self.proc.send_signal(signal.SIGTERM)
            try:
                self.proc.wait(timeout=25)
            except subprocess.TimeoutExpired:
                self.proc.kill()
                self.proc.wait(timeout=10)
        try:
            self.logf.close()
        except Exception:
            pass
        self._wait_port_free()

    def _wait_port_free(self):
        deadline = time.time() + C.PORT_FREE_TIMEOUT_S
        while time.time() < deadline:
            s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            s.settimeout(0.5)
            free = s.connect_ex((C.HOST, C.PORT)) != 0
            s.close()
            if free:
                return
            time.sleep(0.5)


# ----------------------------------------------------------------- think split
THINK_OPEN = re.compile(r"^\s*<think>", re.IGNORECASE)


def split_think(raw: str, is_reasoning: bool):
    """Return (thinking_text, answer_text).

    For reasoning models:
    The chat template opens `<think>` at the end of the prompt, so generation
    always begins *inside* the reasoning block. It emits reasoning, then `</think>`,
    then the final answer. If `</think>` is absent the reasoning was truncated by the
    token cap, so the entire output is thinking (not answer).

    For non-reasoning models:
    If is_reasoning is False, there is no thinking block, so the entire raw text
    is the answer.
    """
    if not is_reasoning:
        return "", raw.strip()

    idx = raw.lower().find("</think>")
    if idx == -1:
        # No closing tag. Because the chat template opens <think> at the end of the
        # prompt, generation begins *inside* the reasoning block and the opening tag
        # never appears in the output — so its absence says nothing. What the missing
        # </think> does say is that the token cap truncated the model mid-reasoning:
        # everything generated is thinking, and there is no answer yet.
        #
        # The previous version returned ("", raw) whenever no literal <think> appeared,
        # which filed an entire truncated reasoning block as the *answer*. That silently
        # corrupted 75 of 210 GGUF speed rows and every MLX speed row.
        return THINK_OPEN.sub("", raw, count=1).strip(), ""
    thinking = THINK_OPEN.sub("", raw[:idx], count=1).strip()
    answer = raw[idx + len("</think>"):].strip()
    return thinking, answer


# --------------------------------------------------------------- one generation
def run_generation(prompt: str, n_predict: int, sampling: tuple = None) -> dict:
    """Stream /completion, capture TTFT + full timings + text.

    `sampling` is (temp, top_p, top_k) from the model's own recommendation — Qwen 3.8
    asks for temp 1.0 where 3.6 asks for 0.6. The MLX arm reads the same value.
    """
    temp, top_p, top_k = sampling or (C.TEMP, C.TOP_P, C.TOP_K)
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": temp,
        "top_p": top_p,
        "top_k": top_k,
        "seed": C.SEED,
        "cache_prompt": False,
        "stream": True,
    }
    data = json.dumps(payload).encode()
    req = urllib.request.Request(C.BASE_URL + "/completion", data=data,
                                 headers={"Content-Type": "application/json"})

    t0 = time.time()
    ttft_ms = None
    pieces = []
    final = {}
    with urllib.request.urlopen(req, timeout=C.STREAM_READ_TIMEOUT_S) as resp:
        for raw_line in resp:
            line = raw_line.decode("utf-8", "replace").strip()
            if not line.startswith("data:"):
                continue
            chunk = json.loads(line[5:].strip())
            content = chunk.get("content", "")
            if content:
                if ttft_ms is None:
                    ttft_ms = (time.time() - t0) * 1000.0
                pieces.append(content)
            if chunk.get("stop"):
                final = chunk
    wall_ms = (time.time() - t0) * 1000.0
    return {
        "raw_text": "".join(pieces),
        "ttft_ms": ttft_ms,
        "wall_ms": wall_ms,
        "final": final,
    }


def extract_timings(final: dict) -> dict:
    """Pull speed + MTP acceptance out of the final chunk's timings.

    NOTE: the timings' drafted-token counters are named draft_tokens_* here so they
    never collide with the config sweep variable rec['draft_n'] (the --spec-draft-n-max).
    """
    t = final.get("timings", {}) or {}
    # draft acceptance keys vary across builds; probe several.
    d_total = t.get("draft_n", t.get("n_draft", final.get("draft_n")))
    d_acc = t.get("draft_n_accepted", t.get("n_draft_accepted",
              t.get("n_accept", final.get("draft_n_accepted"))))
    acc_rate = None
    if isinstance(d_total, (int, float)) and d_total and isinstance(d_acc, (int, float)):
        acc_rate = d_acc / d_total
    return {
        "timings_raw": t,
        "prompt_n": t.get("prompt_n"),
        "prompt_per_second": t.get("prompt_per_second"),
        "predicted_n": t.get("predicted_n", final.get("tokens_predicted")),
        "predicted_per_second": t.get("predicted_per_second"),
        "draft_tokens_total": d_total,
        "draft_tokens_accepted": d_acc,
        "acceptance_rate": acc_rate,
    }


# ------------------------------------------------------------------- run record
def done_keys(path: str) -> set:
    """Completed-run keys for the GGUF arm.

    Keyed on tier and KV type as well as the sweep axes: the same question at
    shallow/agent/deep, or at fp16 vs q8_0 KV, are different measurements. Without
    those in the key a restarted sweep would append duplicates that then get averaged
    into the summary.
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
            if (r.get("error") is None and r.get("runtime") in (None, "gguf")
                    and not r.get("smoke")):
                seen.add((r.get("phase"), r.get("pass"), r.get("model", "qwen3.6-27b"),
                          r.get("quant"), r.get("draft_n"), r.get("question_id"),
                          r.get("prompt_tier", "shallow"), r.get("kv_quant")))
    return seen


def append_record(rec: dict):
    with open(C.RESULTS_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def config_label(draft_n: int) -> str:
    return "off" if draft_n == 0 else f"mtp{draft_n}"


# ------------------------------------------------------------------------- main
def run(model_id, model_cfg, passes, quants, draft_ns, question_ids, n_predict, phase,
        tier="shallow", kv_quant=None, built=None, smoke=False):
    ensure_dirs()
    seen = done_keys(C.RESULTS_JSONL)
    qs = [QUESTION_BY_ID[qid] for qid in question_ids]
    ctx = C.ctx_for_tier(tier, n_predict)
    sampling = C.sampling_for(model_cfg)

    total = len(passes) * len(quants) * len(draft_ns) * len(qs)
    done = 0
    t_start = time.time()
    log(f"START model={model_id} phase={phase} tier={tier} kv={kv_quant or 'fp16'} ctx={ctx}: "
        f"{total} generations (passes={passes} quants={quants} draft_ns={draft_ns} cap={n_predict})")

    for p in passes:
        for quant in quants:
            for dn in draft_ns:
                pending = [q for q in qs
                           if (phase, p, model_id, quant, dn, q["id"], tier, kv_quant) not in seen]
                if not pending:
                    done += len(qs)
                    log(f"skip [{phase}] {model_id} {quant} {config_label(dn)} pass{p} (already done)")
                    continue

                model_path = model_cfg["quants"][quant]
                sidecar = (model_cfg.get("draft_sidecars") or {}).get(quant)
                srv = Server(model_id, quant, dn, model_path, model_cfg["reasoning_format"],
                             ctx=ctx, kv_quant=kv_quant, tier=tier, draft_sidecar=sidecar)
                log(f"launch [{phase}] {model_id} {quant} {config_label(dn)} pass{p} "
                    f"tier={tier} kv={kv_quant or 'fp16'} ...")
                if not srv.start():
                    for q in pending:
                        append_record({
                            "phase": phase, "pass": p, "model": model_id, "quant": quant, "draft_n": dn,
                            "config": config_label(dn), "question_id": q["id"],
                            "category": q["category"], "error": "server_start_failed",
                            "ts": datetime.now(timezone.utc).isoformat(),
                        })
                        done += 1
                    srv.stop()
                    continue

                for q in pending:
                    user_text = built[q["id"]][tier] if built else q["user"]
                    messages = [
                        {"role": "system", "content": q["system"]},
                        {"role": "user", "content": user_text},
                    ]
                    rec = {
                        "phase": phase, "pass": p, "model": model_id, "quant": quant, "draft_n": dn,
                        "config": config_label(dn), "question_id": q["id"],
                        "category": q["category"], "seed": C.SEED,
                        "n_predict_cap": n_predict, "model_path": model_path,
                        "runtime": "gguf", "prompt_tier": tier,
                        "kv_quant": kv_quant, "ctx": ctx,
                        "temp": sampling[0], "top_p": sampling[1], "top_k": sampling[2],
                        "draft_sidecar": sidecar,
                        # See bench_mlx.py: smoke rows use a short cap and one question,
                        # so they are tagged and excluded from resume keys and reports.
                        "smoke": True if smoke else None,
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "error": None,
                    }
                    try:
                        prompt = apply_template(messages)
                        # Recorded so the MLX arm's locally-rendered prompt can be
                        # compared byte-for-byte against what llama-server's --jinja
                        # produced from the same template. Qwen 3.8 added a
                        # reasoning_effort knob; a mismatch there changes how much the
                        # model thinks without changing anything else visible.
                        rec["prompt_sha256"] = hashlib.sha256(prompt.encode()).hexdigest()
                        gen = run_generation(prompt, n_predict, sampling=sampling)
                        tim = extract_timings(gen["final"])
                        thinking, answer = split_think(gen["raw_text"], model_cfg["reasoning"])
                        rec.update(tim)
                        rec.update({
                            "ttft_ms": gen["ttft_ms"],
                            "wall_ms": gen["wall_ms"],
                            "thinking_tokens": tokenize_count(thinking),
                            "answer_tokens": tokenize_count(answer),
                            "thinking_text": thinking,
                            "answer_text": answer,
                            "raw_text": gen["raw_text"],
                            "output_sha256": hashlib.sha256(gen["raw_text"].encode()).hexdigest(),
                            "output_len_chars": len(gen["raw_text"]),
                        })
                        tok_s = rec.get("predicted_per_second")
                        acc = rec.get("acceptance_rate")
                        toks_s = f"{tok_s:.1f}" if isinstance(tok_s, (int, float)) else "?"
                        acc_s = f", acc {acc*100:.0f}%" if isinstance(acc, (int, float)) else ""
                        ttft = gen["ttft_ms"]
                        ttft_s = f", ttft {ttft:.0f}ms" if isinstance(ttft, (int, float)) else ""
                        log(f"  ok {model_id} {quant} {config_label(dn)} p{p} {q['id']}: "
                            f"{rec.get('predicted_n')} tok @ {toks_s} tok/s{acc_s}{ttft_s}")
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"
                        log(f"  ERR {model_id} {quant} {config_label(dn)} p{p} {q['id']}: {rec['error']}")
                    append_record(rec)
                    done += 1
                    elapsed = time.time() - t_start
                    rate = done / elapsed if elapsed else 0
                    eta = (total - done) / rate if rate else 0
                    log(f"  progress {done}/{total} "
                        f"({elapsed/60:.0f} min elapsed, ETA {eta/60:.0f} min)")

                srv.stop()

    log(f"DONE run in {(time.time()-t_start)/60:.1f} min")
    consolidate()


def consolidate():
    """Fold results.jsonl into a single results.json array."""
    recs = []
    if os.path.exists(C.RESULTS_JSONL):
        with open(C.RESULTS_JSONL) as f:
            for ln in f:
                try:
                    recs.append(json.loads(ln))
                except Exception:
                    pass
    with open(C.RESULTS_JSON, "w") as f:
        json.dump(recs, f, indent=2)
    log(f"consolidated {len(recs)} records -> {C.RESULTS_JSON}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="qwen3.8-27b", choices=list(C.MODELS_CONFIG.keys()),
                    help="Model to benchmark (default: qwen3.8-27b)")
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end validation run")
    ap.add_argument("--phase", choices=["speed", "full"], default=None,
                    help="run only one phase (default: both, speed then full)")
    ap.add_argument("--quant", default=None, help="restrict to one quant (must exist in model config)")
    ap.add_argument("--tiers", default=None,
                    help="comma-separated prompt-depth tiers, e.g. shallow,agent,deep")
    ap.add_argument("--kv-quant", dest="kv_quant", default=None,
                    help="comma-separated KV cache types, e.g. none,q8_0 (matches MLX --kv-bits)")
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    if args.consolidate_only:
        ensure_dirs()
        consolidate()
        return

    model_id = args.model
    model_cfg = C.MODELS_CONFIG[model_id]

    # Validate/gather quants to run
    quants = [args.quant] if args.quant else model_cfg["quant_order"]
    for q in quants:
        if q not in model_cfg["quants"]:
            print(f"Error: Quant '{q}' is not configured for model '{model_id}'")
            sys.exit(1)
        path = model_cfg["quants"][q]
        if not os.path.exists(path):
            print(f"Warning: Model file for {model_id} ({q}) not found at: {path}")

    if args.smoke:
        smoke_quant = quants[0]
        smoke_draft_ns = [model_cfg["draft_ns"][0]]
        if len(model_cfg["draft_ns"]) > 1:
            smoke_draft_ns.append(model_cfg["draft_ns"][-1])
        run(model_id, model_cfg, [1], [smoke_quant], smoke_draft_ns,
            ["q2_coding"], C.SMOKE_N_PREDICT, phase="speed", smoke=True)
        return

    allq = [q["id"] for q in QUESTIONS]
    draft_ns = model_cfg["draft_ns"]

    # Prompts come from the shared cache so both runtimes send identical bytes.
    built = T.load_cached() if args.tiers or args.kv_quant else None
    tier_names = args.tiers.split(",") if args.tiers else ["shallow"]
    kv_opts = args.kv_quant.split(",") if args.kv_quant else [None]
    kv_opts = [None if k in ("", "none", "fp16") else k for k in kv_opts]

    # Phase 1: speed sweep over the full grid at a short cap.
    if args.phase in (None, "speed"):
        for tier in tier_names:
            for kvq in kv_opts:
                run(model_id, model_cfg, [1], quants, draft_ns, allq, C.SPEED_N_PREDICT,
                    phase="speed", tier=tier, kv_quant=kvq, built=built)
    # Phase 2: full-length generations (off config only) for token totals + judging.
    if args.phase in (None, "full"):
        run(model_id, model_cfg, [1], quants, [0], allq, C.FULL_N_PREDICT, phase="full",
            tier="shallow", kv_quant=kv_opts[-1], built=built)


if __name__ == "__main__":
    main()
