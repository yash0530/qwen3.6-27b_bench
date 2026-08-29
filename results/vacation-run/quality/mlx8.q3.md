# mlx8 — q3 `q3_architecture` (architecture)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=845.2s decode=9.7 tok/s prefill=367.9 tok/s gen_tokens=8192 prompt_tokens=224 finish=length
- thinking chars: 3520

## Prompt

Design the architecture of an overnight research pipeline for an AI-infrastructure investing workstation. It must: ingest prices, news and earnings for ~12 sectors; evaluate deterministic tripwire risk signals; synthesize a ranked, evidence-backed morning digest where every insight traces to a computed number or a dated source; optionally have an LLM narrate (never fabricate) the already-true digest; and run on a cron scheduler. Cover the data model, the job registry + failure isolation (a failed step must never abort the chain), idempotency, scheduling, where the LLM brain plugs in safely, and how you keep the digest accurate and auditable. Discuss the key trade-offs and failure modes.

## Answer

# Overnight AI-Infrastructure Research Pipeline — Architecture

## 1. Guiding Principles

| Principle | Consequence |
|-----------|-------------|
| **Compute is truth; narration is decoration.** | The LLM never produces a number, a signal, or a conclusion. It only re-words what the deterministic layer already proved. |
| **Everything is append-only and content-addressed.** | Idempotency and auditability fall out for free. No UPDATE, no DELETE. |
| **A job failure degrades, never aborts.** | The 7 AM digest ships with a "degraded" badge if a sector's price feed was late. It never ships empty. |
| **Determinism is the contract.** | Any job, given the same inputs and the same version of its code, must produce byte-identical outputs. The LLM is the one exception, and it is fenced in. |

---

## 2. Data Model

All tables are **append-only**. Primary keys are natural (business) keys; no auto-increment IDs are exposed to the user. A `row_hash` column (SHA-256 of all business columns) enables content-addressing and idempotent upserts.

### 2.1 Reference / Dimension Tables

```sql
-- The ~12 sectors you track. Stable, rarely changes.
CREATE TABLE sector (
    sector_id        TEXT PRIMARY KEY,   -- 'ai_compute', 'cooling', 'power_infra' …
    display_name     TEXT NOT NULL,
    universe_tickers TEXT[] NOT NULL,   -- tickers that define this sector
    benchmarks       TEXT[] NOT NULL,   -- e.g. ['SMH', 'SOXX', 'NVIDIA']
    version          INT NOT NULL,
    effective_from   TIMESTAMPTZ NOT NULL,
    row_hash         TEXT NOT NULL
);

CREATE TABLE instrument (
    ticker           TEXT PRIMARY KEY,
    name             TEXT NOT NULL,
    sector_id        TEXT REFERENCES sector(sector_id),
    currency         TEXT NOT NULL,
    row_hash         TEXT NOT NULL
);
```

### 2.2 Raw Observations (the "leaf" facts)

```sql
-- One row per (ticker, timestamp). Ingested from a price provider.
CREATE TABLE price_observation (
    ticker           TEXT NOT NULL,
    ts               TIMESTAMPTZ NOT NULL,   -- exchange close timestamp
    open             NUMERIC(18,6),
    high             NUMERIC(18,6),
    low              NUMERIC(18,6),
    close            NUMERIC(18,6) NOT NULL,
    volume           BIGINT,
    adj_close        NUMERIC(18,6),
    source           TEXT NOT NULL,          -- 'nasdaq', 'yahoo', 'vendor_X'
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, ts, source),
    row_hash         TEXT NOT NULL
);

-- One row per news article. Ingested from news APIs / RSS.
CREATE TABLE news_item (
    news_id          TEXT NOT NULL,          -- provider-assigned or content hash
    ts_published     TIMESTAMPTZ NOT NULL,
    source           TEXT NOT NULL,          -- 'reuters', 'bloomberg', 'sec_edgar'
    url              TEXT NOT NULL,
    headline         TEXT NOT NULL,
    body             TEXT,
    sector_id        TEXT,                   -- linked by classifier job (nullable)
    tickers_mentioned TEXT[],
    sentiment_score  NUMERIC(5,4),           -- from a *deterministic* lexicon, not an LLM
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (news_id, source),
    row_hash         TEXT NOT NULL
);

-- One row per earnings report (10-K, 10-Q, 8-K earnings item).
CREATE TABLE earnings_report (
    ticker           TEXT NOT NULL,
    fiscal_period    TEXT NOT NULL,          -- '2024-Q4'
    reported_date    DATE NOT NULL,
    revenue          NUMERIC(20,2),
    gross_margin     NUMERIC(8,4),
    operating_margin NUMERIC(8,4),
    eps_actual       NUMERIC(10,4),
    eps_consensus    NUMERIC(10,4),         -- from provider snapshot
    guidance_rev_lo  NUMERIC(20,2),
    guidance_rev_hi  NUMERIC(20,2),
    source           TEXT NOT NULL,
    ingested_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker, fiscal_period, source),
    row_hash         TEXT NOT NULL
);
```

### 2.3 Computed Signals (deterministic, versioned)

```sql
-- Each signal is a pure function of prior tables + a signal version.
-- Example: "AI compute sector down >3σ from 20-day mean in a single session."
CREATE TABLE risk_signal (
    signal_id        TEXT NOT NULL,          -- ULID, deterministic if we want
    signal_type      TEXT NOT NULL,          -- 'volatility_breach', 'earnings_miss', 'news_spike'
    sector_id        TEXT,
    ticker           TEXT,
    ts_observed      TIMESTAMPTZ NOT NULL,   -- when the condition was TRUE
    severity         SMALLINT NOT NULL,      -- 1..5
    value            NUMERIC(18,6) NOT NULL, -- the computed number that triggered it
    threshold        NUMERIC(18,6) NOT NULL, -- the threshold that was crossed
    window_params    JSONB NOT NULL,         -- {'lookback_days': 20, 'sigma': 3.0}
    code_version     TEXT NOT NULL,          -- git SHA of the signal function
    input_hash       TEXT NOT NULL,          -- hash of all input rows consumed
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (signal_id),
    UNIQUE (signal_type, sector_id, ticker, ts_observed, code_version)
    -- ↑ This unique constraint IS the idempotency key.
);

-- Materialised sector-level aggregates (optional, for fast digest assembly)
CREATE TABLE sector_snapshot (
    sector_id        TEXT NOT NULL,
    ts               TIMESTAMPTZ NOT NULL,
    median_pct_chg   NUMERIC(8,4),
    median_volatility NUMERIC(8,4),
    active_signal_count INT,
    input_hash       TEXT NOT NULL,
    PRIMARY KEY (sector_id, ts),
    row_hash         TEXT NOT NULL
);
```

### 2.4 Digest Layer

```sql
-- The digest is a versioned document. Each version is immutable.
CREATE TABLE digest (
    digest_id        TEXT NOT NULL,          -- ULID
    ts_generated     TIMESTAMPTZ NOT NULL,
    ts_for           DATE NOT NULL,          -- "morning of" date
    status           TEXT NOT NULL,          -- 'complete' | 'degraded' | 'failed'
    degraded_reasons TEXT[],                 -- which inputs were missing
    entry_count      INT NOT NULL,
    input_hash       TEXT NOT NULL,          -- hash of all upstream rows consumed
    PRIMARY KEY (digest_id)
);

-- One row per insight in the digest. This is the AUDIT UNIT.
CREATE TABLE digest_entry (
    entry_id         TEXT NOT NULL,          -- 'E-2025-01-15-003'
    digest_id        TEXT NOT NULL REFERENCES digest(digest_id),
    rank             INT NOT NULL,           -- 1 = most important
    sector_id        TEXT,
    headline         TEXT NOT NULL,          -- deterministic template, NOT LLM
    body             TEXT NOT NULL,          -- deterministic template with numbers filled in
    severity         SMALLINT NOT NULL,
    evidence_refs    JSONB NOT NULL,
    -- evidence_refs example:
    -- [
    --   {"table":"risk_signal","pk":{"signal_id":"01H..."}}
    --   ,{"table":"price_observation","pk":{"ticker":"NVDA","ts":"2025-01-14T20:00:00Z","source":"nasdaq"}}
    --   ,{"table":"news_item","pk":{"news_id":"REU-20250114-4471","source":"reuters"}}
    -- ]
    computed_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (entry_id)
);

-- LLM narration: a SEPARATE document that references digest entries.
CREATE TABLE llm_narration (
    narration_id     TEXT NOT NULL,
    digest_id        TEXT NOT NULL REFERENCES digest(digest_id),
    model            TEXT NOT NULL,          -- 'claude-sonnet-4-20250514'
    prompt_version   TEXT NOT NULL,          -- git SHA of the prompt template
    output_text      TEXT NOT NULL,          -- the narrated prose
    validation_status TEXT NOT NULL,         -- 'pass' | 'fail_hallucination' | 'fail_ref'
    failed_refs      JSONB,                  -- entry_ids that the LLM referenced but don't exist
    generated_at     TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (narration_id)
);
```

### 2.5 Job Execution Log (the "who did what, when, with what inputs" audit trail)

```sql
CREATE TABLE job_run (
    run_id           TEXT NOT NULL,          -- ULID
    job_name         TEXT NOT NULL,
    dag_trigger_id   TEXT,                   -- which cron invocation started this
    status           TEXT NOT NULL,          -- 'success' | 'failed' | 'skipped' | 'timeout'
    started_at       TIMESTAMPTZ,
    finished_at      TIMESTAMPTZ,
    input_hash       TEXT,                   -- what data it consumed
    output_hash      TEXT,                   -- what it produced
    error_message    TEXT,
    retry_count      INT NOT NULL DEFAULT 0,
    PRIMARY KEY (run_id)
);
```

---

## 3. Job Registry and DAG

### 3.1 Job Inventory

```
ingest_prices          →  price_observation
ingest_news            →  news_item
ingest_earnings        →  earnings_report
classify_news_sectors  →  updates news_item.sector_id (INSERT new version)
compute_risk_signals   →  risk_signal
build_sector_snapshots →  sector_snapshot
rank_and_template      →  digest, digest_entry       ← THE CORE JOB
llm_narrate            →  llm_narration
validate_narration     →  updates llm_narration.validation_status
deliver                →  email / Slack / filesystem write
```

### 3.2 DAG (topological order)

```
                    ┌─────────────────┐
                    │  ingest_prices  │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  ingest_news    │
                    └────────┬────────┘
                             │
                    ┌────────┴────────┐
                    │  ingest_earnings│
                    └────────┬────────┘
                             │
              ┌──────────────┼──────────────┐
              │              │              │
     ┌────────┴──────┐  ┌───┴────────┐  ┌──┴───────────────┐
     │classify_news_  │  │compute_    │  │build_sector_     │
     │sectors         │  │risk_signals│  │snapshots         │
     └────────┬──────┘  └───┬────────┘  └──┬───────────────┘
              │              │              │
              └──────────────┼──────────────┘
                             │
                    ┌────────┴────────┐
                    │rank_and_template│  ← deterministic, pure
                    └────────┬────────┘
                             │
              ┌──────────────┴──────────────┐
              │                             │
     ┌────────┴────────┐           ┌────────┴────────┐
     │  llm_narrate    │           │  deliver        │
     └────────┬────────┘           └─────────────────┘
              │
     ┌────────┴────────┐
     │validate_        │
     │narration        │
     └────────┬────────┘
              │
     ┌────────┴────────┐
     │  deliver_final  │  (serves LLM if valid, else raw digest)
     └─────────────────┘
```

### 3.3 Failure Isolation Protocol

The orchestrator (a lightweight Python process or a single Node.js script) walks the DAG in topological order:

```python
def execute_dag(trigger_id):
    results = {}
    for job in topo_sort(ALL_JOBS):
        # 1. Check dependencies
        missing_deps = [
            d for d in job.dependencies
            if results.get(d) in ("failed", "skipped", "timeout")
        ]
        if missing_deps and job.critical_deps:  # e.g., ingest_prices is critical
            results[job.name] = "skipped"
            log_job_run(job, "skipped", f"deps failed: {missing_deps}")
            continue

        # 2. Check idempotency: has this exact (job, input_hash) already succeeded?
        input_hash = compute_input_hash(job)
        if job_already_succeeded(job.name, input_hash):
            results[job.name] = "success_cached"
            continue

        # 3. Execute with timeout and retry
        try:
            with timeout(job.timeout_s):
                for attempt in range(job.max_retries + 1):
                    output = job.run(input_hash)
                    write_output(job, output)
                    log_job_run(job, "success", output_hash=hash(output))
                    break
                else:
                    raise JobExhausted(job.name)
            results[job.name] = "success"

        except Exception as e:
            log_job_run(job, "failed", error=str(e))
            results[job.name] = "failed"
            # CRITICAL: we do NOT raise. The loop continues.
            # Downstream jobs see this as "skipped" or run in degraded mode.

    # 4. The digest job (rank_and_template) is designed to accept
    #    partial inputs. It checks which sectors have data and
    #    marks the digest as 'degraded' with reasons.
    return results
```

**Key design decision:** `rank_and_template` does NOT require all 12 sectors to have complete data. It accepts a `degraded_input` flag per sector and produces a digest with:
- Full entries for sectors that have complete data
- A "DATA UNAVAILABLE" placeholder for sectors whose ingest failed
- A `degraded_reasons` array on the digest row

This means **a single vendor API being down never prevents you from getting your morning brief**.

---

## 4. Idempotency

Idempotency is achieved through **three layers**:

### Layer 1: Content-Addressed Inputs

Every job computes an `input_hash` = SHA-256 of all rows it reads (identified by their `row_hash` values). Before executing, it checks:

```sql
SELECT 1 FROM job_run
WHERE job_name = :name
  AND input_hash = :hash
  AND status = 'success'
LIMIT 1;
```

If a row exists → skip. This makes re-running the entire DAG at any time a no-op.

### Layer 2: Natural-Key Uniqueness

All fact tables have `PRIMARY KEY` or `UNIQUE` constraints on natural keys. Ingest uses `INSERT ... ON CONFLICT DO NOTHING`. If the same price bar arrives twice (e.g., a retry from the provider), the second write is silently ignored.

### Layer 3: Deterministic Signal Generation

Signal functions are **pure**: `(input_rows, code_version) → output_row`. The `UNIQUE (signal_type, sector_id, ticker, ts_observed, code_version)` constraint means re-computing a signal that already exists is a no-op. If you bump `code_version` (e.g., change the σ threshold from 3.0 to 2.5), you get NEW signal rows alongside the old ones. The old rows are never deleted.

**What idempotency does NOT give you:** the LLM step. The LLM is non-deterministic. This is handled in §6.

---

## 5. Scheduling

### 5.1 Cron Topology

```
# /etc/cron.d/research-pipeline  (or systemd timer, or a single cron entry)

# 06:00 UTC — trigger the full DAG (after US market close data is available)
0 6 * * 1-5   /usr/local/bin/pipeline/trigger.sh --dag=morning --tz=UTC

# 06:05 UTC — SLA watchdog: if the DAG hasn't finished in 5 minutes, alert
5 6 * * 1-5   /usr/local/bin/pipeline/sla_check.sh --dag=morning --timeout=300

# 09:00 UTC — optional: pre-market data refresh (futures, overnight Asia)
0 9 * * 1-5   /usr/local/bin/pipeline/trigger.sh --dag=premarket

# Sunday 03:00 UTC — weekly: recompute historical signal baselines
0 3 * * 0     /usr/local/bin/pipeline/trigger.sh --dag=weekly_rebase
```

### 5.2 Trigger Script

`trigger.sh` does:
1. Creates a `dag_trigger` row (ID, timestamp, dag name, status='running')
2. Spawns the Python orchestrator as a subprocess (or via `systemd-run` for isolation)
3. The orchestrator writes its final status back to `dag_trigger`

### 5.3 Backfill and Catch-Up

If the 06:00 run fails entirely (host was down), the **next** day's 06:00 run detects the missing digest and can:
- Re-run the full DAG (idempotency makes this safe)
- Or, more surgically, run only the failed steps (the orchestrator queries `job_run` for gaps)

A `backfill` CLI command makes this manual:
```bash
pipeline backfill --from 2025-01-10 --to 2025-01-12 --dag morning
```

### 5.4 Scheduling Trade-offs

| Option | Pro | Con |
|--------|-----|-----|
| Bare `cron` + shell script | Zero dependencies, trivially debuggable | No built-in retry, no DAG awareness |
| Airflow / Prefect | DAG, retries, UI, backfill built-in | Heavy (Postgres + workers), overkill for 10 jobs |
| **Custom lightweight orchestrator (chosen)** | Full control, ~200 LOC, no infra tax | You own the retry/SLA logic |

For a single-user workstation with ~10 jobs, the custom orchestrator is the right call. You get failure isolation and idempotency without the operational weight of Airflow.

---

## 6. LLM Integration — The Safety Fence

### 6.1 Position in the Pipeline

The LLM is the **last optional step** before delivery. It operates on the `digest_entry` rows that `rank_and_template` has already produced. It has **no access** to raw prices, news, or signals. It sees only:

```json
{
  "digest_id": "d-2025-01-15",
  "entries": [
    {
      "entry_id": "E-2025-01-15-001",
      "sector": "ai_compute",
      "severity": 4,
      "headline": "NVDA 1-day return -5.2% breaches -3σ threshold",
      "body": "NVIDIA closed at $1,847.30, a -5.2% move. The 20-day trailing σ is 2.1%, so the move is -2.47σ. The 20-day mean is $1,922.80. Volume was 182M shares vs. 20-day average of 114M (+59%).",
      "evidence_refs": [
        {"table": "risk_signal", "pk": {"signal_id": "01H..."}},
        {"table": "price_observation", "pk": {"ticker": "NVDA", "ts": "2025-01-14T20:00:00Z", "source": "nasdaq"}}
      ]
    },
    { "entry_id": "E-2025-01-15-002", "...": "..." }
  ]
}
```

### 6.2 Prompt Design (simplified)

```
You are a financial brief narrator. You will receive a set of pre-computed
digest entries. Your job is to write a 300-word morning brief that:

1. Opens with the single most important development.
2. Groups related entries by theme (e.g., "compute hardware", "power").
3. Closes with a one-line "watchlist" for the day.

HARD CONSTRAINTS:
- You may ONLY reference facts present in the provided entries.
- You must cite entries inline as [E-XXXX].
- You must NOT introduce any number, ticker, company name, or claim
  that is not already in the provided text.
- If you are uncertain whether a fact is supported, omit it.

Output plain text. No markdown.
```

### 6.3 Post-Generation Validation (the `validate_narration` job)

This is a **deterministic** job. It runs a rule-based check on the LLM output:

```python
def validate_narration(narration_text, entries):
    failures = []
    entry_ids = {e.entry_id for e in entries}

    # Check 1: Every [E-XXXX] reference in the text must exist
    cited = re.findall(r'\[E-(\d{4}-\d{2}-\d{2}-\d{3})\]', narration_text)
    for ref in cited:
        full_id = f"E-{ref}"
        if full_id not in entry_ids:
            failures.append({"type": "dangling_ref", "ref": full_id})

    # Check 2: No dollar amounts, percentages, or tickers in the narration
    #           that don't appear in the source entries.
    source_text = " ".join(e.body for e in entries)
    source_numbers = extract_numbers(source_text)
    narration_numbers = extract_numbers(narration_text)
    for n in narration_numbers - source_numbers:
        failures.append({"type": "fabricated_number", "value": n})

    # Check 3: No ticker symbols in narration that aren't in source
    source_tickers = extract_tickers(source_text)
    narration_tickers = extract_tickers(narration_text)
    for t in narration_tickers - source_tickers:
        failures.append({"type": "fabricated_ticker", "value": t})

    if failures:
        return "fail", failures
    return "pass", []
```

If validation **fails**, the narration is discarded (`validation_status = 'fail_hallucination'`) and the `deliver_final` job serves the **raw deterministic digest** instead. The user never sees a hallucinated sentence.

### 6.4 Why This Architecture Is Safe

| Attack / Failure | Mitigation |
|-----------------|-----------|
| LLM invents a number | Check 2 catches it; narration discarded |
| LLM references a non-existent entry | Check 1 catches it |
| LLM introduces a new ticker | Check 3 catches it |
| LLM model is down / times out | `llm_narrate` job fails → `deliver_final` serves raw digest |
| LLM model produces garbage (format) | Regex checks fail → raw digest served |
| Prompt injection via news body | The LLM never sees raw news bodies; it only sees pre-templated `digest_entry.body` strings that were produced by deterministic code |

The last point is critical: **the LLM's input is a closed, pre-validated set of strings.** A malicious or anomalous news headline cannot reach the LLM because the deterministic layer must first parse it, compute a signal, and template it into a `digest_entry` with evidence refs. The news body is in `news_item.body`, which the LLM never reads.

---

## 7. Auditability — The "Click Any Claim" Invariant

The system guarantees: **any sentence in the delivered digest can be traced to a row in a fact table within two hops.**

```
User reads: "NVDA closed at $1,847.30, a -5.2% move [E-2025-01-15-001]"
                │
                ▼  (Hop 1: look up entry)
digest_entry E-2025-01-15-001
  evidence_refs → risk_signal 01H..., price_observation (NVDA, 2025-01-14T20:00Z, nasdaq)
                │
                ▼  (Hop 2: look up evidence)
price_observation row:
  close = 1847.30, source = 'nasdaq', ingested_at = 2025-01-14T23:45:01Z
```

A simple CLI or web endpoint makes this trivial:

```bash
$ pipeline prove E-2025-01-15-001
[1/2] digest_entry E-2025-01-15-001
      headline: "NVDA 1-day return -5.2% breaches -3σ threshold"
      evidence: 2 refs
[2/2] Evidence row 1: risk_signal 01H...
      type: volatility_breach, value: -5.2%, threshold: -3.0σ
      input_hash: a3f2...
      code_version: 4b7c91d (signals/volatility_breach.py @ main)
[2/2] Evidence row 2: price_observation (NVDA, 2025-01-14T20:00:00Z, nasdaq)
      close: 1847.30, volume: 182_400_000
      ingested_at: 2025-01-14T23:45:01Z
```

Because every table is append-only and hash-chained, **you can replay the entire pipeline from any point in time** and verify that the digest you received on a given morning was the only possible output given the inputs that existed at that time.

---

## 8. Key Trade-offs

| Decision | Chose | Alternative | Why |
|----------|-------|-------------|-----|
| **Orchestration** | Custom ~200 LOC Python | Airflow / Prefect | 10 jobs on one box; Airflow's Postgres + worker + web UI is 10× the ops cost for zero benefit |
| **LLM position** | Post-digest narration only | LLM as an analyst (generates insights) | The moment the LLM can generate insights, you lose the auditability guarantee. The fence in §6 is necessary but not sufficient; the architectural choice to make the LLM a formatter is the real safety |
| **Storage** | Single Postgres instance | Parquet files + DuckDB | You need ACID for the `job_run` table and for the `INSERT ON CONFLICT` idempotency. At this scale (~100K rows/year), Postgres is simpler and the performance is irrelevant |
| **Signal computation** | In-process Python functions | Separate microservices / Kafka streams | Determinism and simplicity matter more than throughput. The signals are computed once per day, not per tick |
| **News classification** | Deterministic keyword/regex rules | LLM-based classification | The classifier's output feeds the deterministic signal layer. If the classifier is non-deterministic, the signals are non-deterministic, and idempotency breaks. A 90%-accurate regex is better than a 99%-accurate LLM that occasionally flips |
| **Delivery** | Email + filesystem | Slack bot, Telegram, etc. | Email is auditable (you have the .eml), has no API rate limits, and doesn't require a token. The filesystem copy is your "ground truth" archive |

---

## 9. Failure Modes and Mitigations

| Failure | Impact | Mitigation |
|---------|--------|-----------|
| **Price API returns stale data** (e.g., yesterday's close instead of today's) | Signals computed on wrong data; digest is "correct" but meaningless | Validate: `ts` in response must be within expected window. If not, reject the batch and mark ingest as failed. The digest degrades for that sector. |
| **News API returns a malformed JSON** | `ingest_news` fails for that batch | Retry with backoff (3×). If still failing, mark as failed. Signals that depend on news (e.g., "news_spike") won't fire. Digest degrades. |
| **Disk full** | All writes fail | Monitor disk usage in the cron job; alert at 80%. Use `pg_size` checks. At this scale, data grows ~50 MB/year, so this is unlikely but cheap to guard against. |
| **LLM provider is down** | Narration unavailable | `llm_narrate` job times out → `deliver_final` serves raw digest. No user-facing impact beyond "no narrative prose." |
| **LLM hallucinates** | Invalid numbers in output | Validation job catches it (§6.3). Narration discarded. Raw digest served. |
| **A signal function has a bug** (e.g., off-by-one in σ calculation) | All signals of that type are wrong | `code_version` (git SHA) is stored on every signal row. You can identify the exact version that produced a given signal and re-run with a fix. Old rows remain (append-only) so you can compare. |
| **Clock skew / timezone bug** | Signals computed with wrong "today" | All timestamps are stored in UTC. The `ts_for` field on the digest is a `DATE` in UTC. The cron job runs at a fixed UTC time. No local-timezone logic anywhere. |
| **The host machine reboots at 05:58** | The 06:00 cron never fires | The 06:05 SLA watchdog (a *separate* cron entry on a different host, or a systemd timer with `OnCalendar` + `Persistent=true`) fires, detects no `dag_trigger` row for today, and re-triggers the DAG. Idempotency makes this safe. |
| **Someone edits a signal function and re-runs** | New signal rows appear alongside old ones | `code_version` distinguishes them. The digest job reads the **latest** `code_version` by default but can be pinned. The `job_run` table records who ran what when. |

---

## 10. Operational Checklist (Morning)

```
05:55  Price vendor publishes close data
06:00  Cron fires → orchestrator starts DAG
06:01  ingest_prices:  12 sectors × ~50 tickers = ~600 rows inserted
06:02  ingest_news:    ~200 items (filtered to relevant sources)
06:02  ingest_earnings: ~5 new 8-Ks this quarter
06:03  classify_news_sectors: 200 items tagged
06:04  compute_risk_signals: 7 new signals (2 severity-4, 1 severity-5)
06:04  build_sector_snapshots: 12 rows
06:05  rank_and_template: 12 entries, ranked by severity × sector_importance
06:06  llm_narrate: 320 words, 12 [E-...] citations
06:06  validate_narration: PASS
06:07  deliver
