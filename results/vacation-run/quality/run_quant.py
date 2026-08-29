#!/usr/bin/env python3
"""Run the 6 quality questions against the resident MLX server and save answers.

Usage: run_quant.py <quant-alias> <model-path>

Safety: aborts immediately if kIOGPUCommandBufferCallbackErrorOutOfMemory appears
in the server log, or if a new kernel-panic file shows up.
"""
import json, os, sys, time, urllib.request, urllib.error, glob, re

QDIR = os.path.dirname(os.path.abspath(__file__))
LOG = os.path.expanduser("~/.local/state/local-llm/logs/server.log")
PANIC_GLOB = "/Library/Logs/DiagnosticReports/panic-full-*.panic"
PANIC_BASELINE = 1
ENDPOINT = "http://127.0.0.1:8089/v1/chat/completions"

SYSTEM = (
    "You are a senior staff engineer and AI-infrastructure research analyst acting as "
    "the reasoning brain of a personal investing-research workstation. Reason carefully "
    "and then give a precise, well-structured, technically deep answer."
)

TEMP = 1.0        # Qwen3.8 thinking-mode recommendation; llm-serve model_temp(mlx*) = 1.0
TOP_P = 0.95
MAX_TOKENS = 8192
SEED = 1234
EFFORT = "low"   # 4096+xhigh spent the entire budget on reasoning (0 answer chars)


def guard():
    n = len(glob.glob(PANIC_GLOB))
    if n > PANIC_BASELINE:
        sys.exit(f"ABORT: panic file count {n} > baseline {PANIC_BASELINE}")
    try:
        with open(LOG, "r", errors="replace") as f:
            body = f.read()
    except FileNotFoundError:
        return
    if "kIOGPUCommandBufferCallbackErrorOutOfMemory" in body:
        sys.exit("ABORT: GPU OOM in server log")


def log_size():
    try:
        return os.path.getsize(LOG)
    except OSError:
        return 0


def decode_stats(since):
    """Parse the last 'Request completed' line written after byte offset `since`."""
    try:
        with open(LOG, "r", errors="replace") as f:
            f.seek(since)
            tail = f.read()
    except OSError:
        return {}
    lines = [l for l in tail.splitlines() if "Request completed:" in l]
    if not lines:
        return {}
    l = lines[-1]
    out = {}
    for k, pat in (("prompt_tokens", r"prompt_tokens=(\d+)"),
                   ("generated_tokens", r"generated_tokens=(\d+)"),
                   ("elapsed_s", r"elapsed=([\d.]+)s"),
                   ("prefill_tps", r"prefill=([\d.]+) tok/s"),
                   ("decode_tps", r"decode=([\d.]+) tok/s"),
                   ("finish_reason", r"finish_reason=(\S+)")):
        m = re.search(pat, l)
        if m:
            out[k] = m.group(1)
    return out


def ask(model, q):
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q["user"]},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMP,
        "top_p": TOP_P,
        "seed": SEED,
        "stream": False,
        "enable_thinking": True,
        "reasoning_effort": EFFORT,
        "chat_template_kwargs": {"enable_thinking": True},
    }
    req = urllib.request.Request(
        ENDPOINT, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"})
    t0 = time.time()
    with urllib.request.urlopen(req, timeout=1800) as r:
        body = json.loads(r.read().decode())
    return body, time.time() - t0


def main():
    quant, model = sys.argv[1], sys.argv[2]
    questions = json.load(open(os.path.join(QDIR, "questions.json")))
    summary = []
    for i, q in enumerate(questions, 1):
        guard()
        off = log_size()
        print(f"[{time.strftime('%H:%M:%S')}] {quant} q{i} {q['id']} ...", flush=True)
        try:
            body, wall = ask(model, q)
        except Exception as e:
            print(f"  FAILED: {e}", flush=True)
            guard()
            summary.append({"q": i, "id": q["id"], "error": str(e)})
            continue
        guard()
        msg = body["choices"][0]["message"]
        content = msg.get("content") or ""
        thinking = msg.get("reasoning_content") or msg.get("reasoning") or ""
        usage = body.get("usage", {})
        st = decode_stats(off)
        rec = {"q": i, "id": q["id"], "wall_s": round(wall, 1),
               "decode_tps": st.get("decode_tps"), "prefill_tps": st.get("prefill_tps"),
               "gen_tokens": st.get("generated_tokens") or usage.get("completion_tokens"),
               "prompt_tokens": st.get("prompt_tokens") or usage.get("prompt_tokens"),
               "finish_reason": st.get("finish_reason"),
               "think_chars": len(thinking), "answer_chars": len(content)}
        summary.append(rec)
        print(f"  {rec}", flush=True)

        out = os.path.join(QDIR, f"{quant}.q{i}.md")
        with open(out, "w") as f:
            f.write(f"# {quant} — q{i} `{q['id']}` ({q['category']})\n\n")
            f.write(f"- temp={TEMP} top_p={TOP_P} max_tokens={MAX_TOKENS} seed={SEED} "
                    f"thinking=on effort={EFFORT} APC=0 SPEC=0\n")
            f.write(f"- wall={rec['wall_s']}s decode={rec['decode_tps']} tok/s "
                    f"prefill={rec['prefill_tps']} tok/s gen_tokens={rec['gen_tokens']} "
                    f"prompt_tokens={rec['prompt_tokens']} finish={rec['finish_reason']}\n")
            f.write(f"- thinking chars: {len(thinking)}\n\n")
            f.write("## Prompt\n\n")
            f.write(q["user"] + "\n\n")
            f.write("## Answer\n\n")
            f.write(content.strip() + "\n")
        print(f"  -> {out}", flush=True)

    with open(os.path.join(QDIR, f"{quant}.stats.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print("DONE", flush=True)


if __name__ == "__main__":
    main()
