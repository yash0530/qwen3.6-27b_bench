#!/usr/bin/env python3
"""Benchmark orchestrator for Qwen3.6-27B MTP quants.

For every (pass, quant, draft_n) it launches a dedicated llama-server, runs the 5
questions, and records rich per-run metrics to results/results.jsonl (crash-safe,
resumable). Pure stdlib — no third-party deps.

Usage:
  python3 bench.py             # full run: all passes x quants x draft_n x questions
  python3 bench.py --smoke     # tiny end-to-end validation (~3-5 min)
  python3 bench.py --pass 1    # only pass 1
  python3 bench.py --quant q6  # restrict to one quant
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
    def __init__(self, quant: str, draft_n: int):
        self.quant = quant
        self.draft_n = draft_n
        self.model = C.MODELS[quant]
        self.log_path = os.path.join(C.LOGS, f"server_{quant}_n{draft_n}.log")
        self.proc = None

    def start(self):
        cmd = C.server_cmd(self.model, self.draft_n, self.log_path)
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


def split_think(raw: str):
    """Return (thinking_text, answer_text). Qwen3.6 emits reasoning then </think>."""
    idx = raw.lower().find("</think>")
    if idx == -1:
        # No closing tag: either a non-thinking answer or truncated reasoning.
        if THINK_OPEN.search(raw):
            return THINK_OPEN.sub("", raw, count=1).strip(), ""
        return "", raw.strip()
    thinking = raw[:idx]
    thinking = THINK_OPEN.sub("", thinking, count=1).strip()
    answer = raw[idx + len("</think>"):].strip()
    return thinking, answer


# --------------------------------------------------------------- one generation
def run_generation(prompt: str, n_predict: int) -> dict:
    """Stream /completion, capture TTFT + full timings + text."""
    payload = {
        "prompt": prompt,
        "n_predict": n_predict,
        "temperature": C.TEMP,
        "top_p": C.TOP_P,
        "top_k": C.TOP_K,
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
    seen = set()
    if not os.path.exists(path):
        return seen
    with open(path) as f:
        for ln in f:
            try:
                r = json.loads(ln)
            except Exception:
                continue
            if r.get("error") is None:
                seen.add((r["pass"], r["quant"], r["draft_n"], r["question_id"]))
    return seen


def append_record(rec: dict):
    with open(C.RESULTS_JSONL, "a") as f:
        f.write(json.dumps(rec) + "\n")


def config_label(draft_n: int) -> str:
    return "off" if draft_n == 0 else f"mtp{draft_n}"


# ------------------------------------------------------------------------- main
def run(passes, quants, draft_ns, question_ids, n_predict):
    ensure_dirs()
    seen = done_keys(C.RESULTS_JSONL)
    qs = [QUESTION_BY_ID[qid] for qid in question_ids]

    total = len(passes) * len(quants) * len(draft_ns) * len(qs)
    done = 0
    t_start = time.time()
    log(f"START run: {total} generations "
        f"(passes={passes} quants={quants} draft_ns={draft_ns} n_predict={n_predict})")

    for p in passes:
        for quant in quants:
            for dn in draft_ns:
                pending = [q for q in qs if (p, quant, dn, q["id"]) not in seen]
                if not pending:
                    done += len(qs)
                    log(f"skip {quant} {config_label(dn)} pass{p} (already done)")
                    continue

                srv = Server(quant, dn)
                log(f"launch {quant} {config_label(dn)} pass{p} ...")
                if not srv.start():
                    for q in pending:
                        append_record({
                            "pass": p, "quant": quant, "draft_n": dn,
                            "config": config_label(dn), "question_id": q["id"],
                            "category": q["category"], "error": "server_start_failed",
                            "ts": datetime.now(timezone.utc).isoformat(),
                        })
                        done += 1
                    srv.stop()
                    continue

                for q in pending:
                    messages = [
                        {"role": "system", "content": q["system"]},
                        {"role": "user", "content": q["user"]},
                    ]
                    rec = {
                        "pass": p, "quant": quant, "draft_n": dn,
                        "config": config_label(dn), "question_id": q["id"],
                        "category": q["category"], "seed": C.SEED,
                        "model_path": C.MODELS[quant],
                        "ts": datetime.now(timezone.utc).isoformat(),
                        "error": None,
                    }
                    try:
                        prompt = apply_template(messages)
                        gen = run_generation(prompt, n_predict)
                        tim = extract_timings(gen["final"])
                        thinking, answer = split_think(gen["raw_text"])
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
                        log(f"  ok {quant} {config_label(dn)} p{p} {q['id']}: "
                            f"{rec.get('predicted_n')} tok @ {toks_s} tok/s{acc_s}{ttft_s}")
                    except Exception as e:
                        rec["error"] = f"{type(e).__name__}: {e}"
                        log(f"  ERR {quant} {config_label(dn)} p{p} {q['id']}: {rec['error']}")
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
    ap.add_argument("--smoke", action="store_true", help="tiny end-to-end validation run")
    ap.add_argument("--pass", dest="passes", type=int, choices=[1, 2], default=None)
    ap.add_argument("--quant", choices=list(C.MODELS), default=None)
    ap.add_argument("--consolidate-only", action="store_true")
    args = ap.parse_args()

    if args.consolidate_only:
        ensure_dirs()
        consolidate()
        return

    if args.smoke:
        run(C.SMOKE_PASSES, C.SMOKE_QUANTS, C.SMOKE_DRAFT_NS,
            C.SMOKE_QUESTION_IDS, C.SMOKE_N_PREDICT)
        return

    passes = [args.passes] if args.passes else C.PASSES
    quants = [args.quant] if args.quant else C.QUANT_ORDER
    run(passes, quants, C.DRAFT_NS, [q["id"] for q in QUESTIONS], C.N_PREDICT)


if __name__ == "__main__":
    main()
