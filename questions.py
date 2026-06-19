"""The 5 benchmark prompts.

Each elicits substantial reasoning so that tokens/sec, thinking-token volume and
MTP acceptance are measured over realistic, long generations. Categories span the
user's actual use case: model selection / market research, coding, architecture,
financial-analysis reasoning, and debugging.
"""

SYSTEM = (
    "You are a senior staff engineer and AI-infrastructure research analyst acting as "
    "the reasoning brain of a personal investing-research workstation. Reason carefully "
    "and then give a precise, well-structured, technically deep answer."
)

QUESTIONS = [
    {
        "id": "q1_model_selection",
        "category": "market_research",
        "system": SYSTEM,
        "user": (
            "I have a MacBook M5 Pro with 64 GB of unified RAM. I want to run the best "
            "open-source model that fits in about 45 GB of RAM, to act as a strong "
            "Sonnet-4.6 replacement AND as the main brain of a financial-analysis research "
            "engine (overnight ETL of price/news/earnings, tripwire risk signals, a "
            "deterministic morning digest that an LLM narrates). Do the market research: "
            "what are my realistic open-weight options in mid-2026, how do they compare on "
            "reasoning, coding, long-context and tool-use, and which one would you pick and "
            "why? Then explain how to host it on this machine to maximize accuracy and "
            "performance (quantization, context, KV cache, speculative/MTP decoding, serving "
            "stack), and how to wire it into the research engine. Be concrete and opinionated."
        ),
    },
    {
        "id": "q2_coding",
        "category": "coding",
        "system": SYSTEM,
        "user": (
            "Implement, in TypeScript, a rate-limited asynchronous job queue suitable for an "
            "overnight ETL pipeline. Requirements: a configurable max concurrency and a max "
            "requests-per-second token-bucket limiter; per-item retry with exponential backoff "
            "and jitter; per-item failure isolation so one failure never aborts the batch; a "
            "result object that counts successes/failures and collects error details; and "
            "graceful cancellation via an AbortSignal. Provide the full implementation with "
            "types, then a short usage example and notes on the trade-offs you made."
        ),
    },
    {
        "id": "q3_architecture",
        "category": "architecture",
        "system": SYSTEM,
        "user": (
            "Design the architecture of an overnight research pipeline for an AI-infrastructure "
            "investing workstation. It must: ingest prices, news and earnings for ~12 sectors; "
            "evaluate deterministic tripwire risk signals; synthesize a ranked, evidence-backed "
            "morning digest where every insight traces to a computed number or a dated source; "
            "optionally have an LLM narrate (never fabricate) the already-true digest; and run "
            "on a cron scheduler. Cover the data model, the job registry + failure isolation "
            "(a failed step must never abort the chain), idempotency, scheduling, where the LLM "
            "brain plugs in safely, and how you keep the digest accurate and auditable. Discuss "
            "the key trade-offs and failure modes."
        ),
    },
    {
        "id": "q4_finance_reasoning",
        "category": "finance_reasoning",
        "system": SYSTEM,
        "user": (
            "It is a Monday morning. Across 12 AI-infrastructure sectors (compute/GPUs, "
            "memory/HBM, networking, foundry, power & cooling, hyperscaler capex, model labs, "
            "inference providers, edge silicon, optical, storage, and EDA), these tripwires "
            "fired overnight: (a) an HBM supplier guided down on yields; (b) a hyperscaler cut "
            "next-quarter capex guidance by 8%; (c) a foundry raised leading-node prices; (d) "
            "two inference providers cut API prices ~30%; (e) a power-utility serving data "
            "centers had a transmission outage. Reason step by step about the second-order "
            "effects, which sectors are most at risk vs. positioned to benefit, what is signal "
            "vs. noise, and how you would adjust research focus and risk attention this week. "
            "Show your chain of reasoning, then a concise prioritized summary."
        ),
    },
    {
        "id": "q5_debugging",
        "category": "debugging",
        "system": SYSTEM,
        "user": (
            "The following TypeScript is meant to cache the result of an async loader so "
            "concurrent callers share one in-flight request, but under load it sometimes makes "
            "duplicate network calls and occasionally caches a rejected promise forever. Find "
            "every bug, explain the race conditions precisely, and provide a corrected, "
            "production-quality version.\n\n"
            "```ts\n"
            "const cache = new Map<string, Promise<Data>>();\n"
            "async function load(key: string): Promise<Data> {\n"
            "  if (cache.has(key)) return cache.get(key)!;\n"
            "  const data = await fetchData(key);\n"
            "  const p = Promise.resolve(data);\n"
            "  cache.set(key, p);\n"
            "  return data;\n"
            "}\n"
            "```"
        ),
    },
]

QUESTION_BY_ID = {q["id"]: q for q in QUESTIONS}
