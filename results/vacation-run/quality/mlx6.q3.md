# mlx6 — q3 `q3_architecture` (architecture)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=665.7s decode=12.3 tok/s prefill=336.9 tok/s gen_tokens=8192 prompt_tokens=224 finish=length
- thinking chars: 4187

## Prompt

Design the architecture of an overnight research pipeline for an AI-infrastructure investing workstation. It must: ingest prices, news and earnings for ~12 sectors; evaluate deterministic tripwire risk signals; synthesize a ranked, evidence-backed morning digest where every insight traces to a computed number or a dated source; optionally have an LLM narrate (never fabricate) the already-true digest; and run on a cron scheduler. Cover the data model, the job registry + failure isolation (a failed step must never abort the chain), idempotency, scheduling, where the LLM brain plugs in safely, and how you keep the digest accurate and auditable. Discuss the key trade-offs and failure modes.

## Answer

# Overnight AI-Infrastructure Research Pipeline — Architecture

## Design Philosophy

The system has one invariant that governs every decision:

> **The LLM is a renderer, not a generator.** It operates exclusively on a fully-computed, provenance-annotated digest. It is architecturally incapable of introducing a fact that does not already exist in the structured input.

Everything else — scheduling, idempotency, failure isolation — serves that invariant and the practical need for a deterministic, reproducible, auditable output.

---

## 1. Data Model

### Core Entities

```
┌─────────────────────────────────────────────────────────────────────┐
│                        DATA MODEL (relational)                       │
├─────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  Sector(id, name, gics_code, is_active)                             │
│  Ticker(id, symbol, name, sector_id, is_active)                     │
│                                                                      │
│  MarketData(                                                       │
│    ticker_id, as_of_ts, close, adj_close, volume,                   │
│    vwap, intraday_high, intraday_low,                               │
│    source, ingestion_ts                                            │
│  )                                                                  │
│  UNIQUE(ticker_id, as_of_ts)                                        │
│                                                                      │
│  NewsItem(                                                         │
│    id, source, url, url_hash, headline,                             │
│    published_ts, ingested_ts, body_md,                              │
│    entities_jsonb  -- [{type: "ticker"|"sector"|"person", ref}]     │
│    sentiment_score, relevance_score                                 │
│  )                                                                  │
│  UNIQUE(source, url_hash)  -- natural dedup                         │
│                                                                      │
│  EarningsReport(                                                   │
│    ticker_id, fiscal_period, reported_at,                           │
│    revenue, eps, ebitda, guidance_jsonb,                            │
│    beat_miss_jsonb  -- {revenue: "beat"|"miss"|"inline", ...},      │
│    source_url, source_publisher                                     │
│  )                                                                  │
│  UNIQUE(ticker_id, fiscal_period)                                   │
│                                                                      │
│  Signal(                                                           │
│    id, run_id, sector_id, type,                                     │
│    triggered_at, value, threshold, direction,                       │
│    window_start, window_end,                                        │
│    context_jsonb  -- the exact inputs that produced this,           │
│    severity  -- {info, watch, alert, critical}                     │
│  )                                                                  │
│  UNIQUE(run_id, sector_id, type, window_start, window_end)          │
│    -- idempotency: re-running the same window produces no dup       │
│                                                                      │
│  Digest(                                                           │
│    run_id, generated_at,                                            │
│    sections_jsonb  -- ordered list of sections, each containing     │
│                       an array of Claim objects (see below)         │
│    narrative_text  -- LLM-rendered prose (nullable until rendered)  │
│    schema_version                                                          │
│  )                                                                  │
│                                                                      │
│  Claim(  -- embedded in Digest.sections_jsonb, not a separate table │
│    id, section, text,                                               │
│    provenance: {                                                    │
│      kind: "computed" | "source" | "derived",                      │
│      ref_type: "signal" | "market_data" | "news" | "earnings",     │
│      ref_id: <uuid or composite key>,                              │
│      computation: <optional: function_name + input hash>            │
│    },                                                               │
│    confidence: {level: "exact"|"derived"|"reported", detail}       │
│  )                                                                  │
│                                                                      │
│  JobRun(                                                           │
│    run_id, job_id, started_at, finished_at,                         │
│    status: {pending, running, success, failed, skipped},           │
│    error_msg, outputs_jsonb,                                        │
│    parent_run_id  -- for nested jobs                                │
│  )                                                                  │
│                                                                      │
└─────────────────────────────────────────────────────────────────────┘
```

### Why `Claim` is embedded, not a table

Claims are only meaningful in the context of a specific digest. They are append-only within a run and never modified. A separate table would complicate queries for no benefit. However, if you later want to cross-reference claims across runs (e.g., "when did this signal first appear?"), you promote it to a materialized view or a secondary index.

### Provenance: the audit backbone

Every claim in the digest carries a **provenance pointer**. This is what makes the digest auditable:

- `computed` → a deterministic function was applied to known inputs. The `computation` field names the function and includes a hash of its inputs, so you can replay it.
- `source` → a news article or earnings report. `ref_id` points to the row.
- `derived` → a transformation of another claim (e.g., "sector aggregate = weighted average of constituents").

A human (or an automated auditor) can always trace from a sentence in the digest back to the exact number or article that produced it.

---

## 2. Job Registry and Failure Isolation

### Job Manifest

Jobs are declared in a version-controlled YAML manifest. This is the single source of truth for the DAG.

```yaml
# jobs/pipeline.yaml
version: 3
schedule: "0 6 * * 1-5"   # 06:00 weekdays

jobs:
  - id: ingest_market_data
    fn: pipelines.ingest.fetch_market_data
    inputs:
      - type: config
        key: sector_tickers
    outputs:
      - type: table
        ref: market_data
      - type: count
        ref: rows_ingested
    timeout_s: 300
    retries: {max: 3, backoff_s: [10, 30, 90]}
    on_failure: degrade   # see failure policy below

  - id: ingest_news
    fn: pipelines.ingest.fetch_news
    inputs:
      - type: config
        key: news_sources
      - type: time_window
        hours_back: 24
    outputs:
      - type: table
        ref: news_items
    timeout_s: 600
    retries: {max: 2, backoff_s: [15, 45]}
    on_failure: degrade

  - id: ingest_earnings
    fn: pipelines.ingest.fetch_earnings
    inputs:
      - type: time_window
        hours_back: 48
    outputs:
      - type: table
        ref: earnings_reports
    timeout_s: 300
    retries: {max: 2, backoff_s: [10, 30]}
    on_failure: degrade

  - id: compute_signals
    fn: pipelines.signals.run_all
    depends_on: [ingest_market_data, ingest_news, ingest_earnings]
    inputs:
      - type: table
        ref: market_data
        optional: true    # can run with partial data
      - type: table
        ref: news_items
        optional: true
      - type: table
        ref: earnings_reports
        optional: true
    outputs:
      - type: table
        ref: signals
    timeout_s: 120
    retries: {max: 1, backoff_s: [30]}
    on_failure: degrade

  - id: compute_sector_aggregates
    fn: pipelines.aggregates.sector_rollup
    depends_on: [compute_signals, ingest_market_data]
    inputs:
      - type: table
        ref: signals
        optional: true
      - type: table
        ref: market_data
        optional: true
    outputs:
      - type: json
        ref: sector_aggregates
    timeout_s: 60

  - id: build_digest
    fn: pipelines.digest.assemble
    depends_on: [compute_signals, compute_sector_aggregates]
    inputs:
      - type: json
        ref: sector_aggregates
        optional: true
      - type: table
        ref: signals
        optional: true
    outputs:
      - type: json
        ref: digest_structure
    timeout_s: 60
    on_failure: abort   # if we can't build the digest, the pipeline failed

  - id: render_narrative
    fn: pipelines.llm.narrate
    depends_on: [build_digest]
    inputs:
      - type: json
        ref: digest_structure
        optional: false
    outputs:
      - type: text
        ref: narrative_text
      - type: json
        ref: validation_report
    timeout_s: 120
    retries: {max: 2, backoff_s: [10, 20]}
    on_failure: degrade  # digest ships without narrative

  - id: deliver
    fn: pipelines.delivery.send
    depends_on: [build_digest, render_narrative]
    inputs:
      - type: json
        ref: digest_structure
        optional: true
      - type: text
        ref: narrative_text
        optional: true
    outputs:
      - type: confirmation
        ref: delivery_status
    timeout_s: 60
    on_failure: alert    # human must know delivery failed
```

### Failure Policy Enum

| Policy | Behavior |
|--------|----------|
| `degrade` | Job is marked `failed`. Downstream jobs that list this output as `optional: true` proceed with a `gap` annotation in the digest. Downstream jobs that list it as `optional: false` are marked `skipped`. |
| `abort` | The entire run is marked `failed`. No partial delivery. An alert fires. |
| `alert` | Same as `degrade` but additionally fires a human-facing alert (PagerDuty/Slack). |

### Execution Model

Each job runs as an **isolated subprocess** (or container, if you want heavier isolation). The scheduler:

1. Resolves the DAG from the manifest.
2. Topologically sorts jobs.
3. For each ready job (all non-optional deps satisfied):
   - Spawns a subprocess with a serialized input payload (via a temp file or a message queue).
   - The subprocess writes its outputs to a run-scoped output directory.
   - On exit code 0 → `success`. Non-zero or timeout → `failed`.
4. After all jobs complete (or are skipped), the run is finalized.

```
Scheduler (long-lived process)
  │
  ├── spawn: ingest_market_data  (pid 12345)
  ├── spawn: ingest_news         (pid 12346)
  ├── spawn: ingest_earnings     (pid 12347)
  │         (these three run in parallel)
  │
  │  wait for all three
  │
  ├── spawn: compute_signals     (pid 12348)
  │
  │  wait
  │
  ├── spawn: compute_sector_aggregates  (pid 12349)
  │
  │  wait
  │
  ├── spawn: build_digest        (pid 12350)
  │
  │  wait
  │
  ├── spawn: render_narrative    (pid 12351)
  │
  │  wait
  │
  └── spawn: deliver            (pid 12352)
```

**Why subprocess isolation?** A segfault, memory leak, or unhandled exception in one job cannot take down the scheduler or other jobs. The scheduler only sees the exit code and the output files.

---

## 3. Idempotency

Idempotency is achieved at three levels:

### Level 1: Natural-key upserts on ingest

```python
def upsert_market_data(rows: list[MarketData], conn):
    for row in rows:
        conn.execute("""
            INSERT INTO market_data (ticker_id, as_of_ts, close, volume, ...)
            VALUES (...)
            ON CONFLICT (ticker_id, as_of_ts) DO UPDATE SET
                close = EXCLUDED.close,
                volume = EXCLUDED.volume,
                ingestion_ts = now()
        """)
```

Re-ingesting the same day's close is a no-op (or an overwrite with the same value). No duplicates.

### Level 2: Signal deduplication

Signals carry a composite natural key: `(run_id, sector_id, type, window_start, window_end)`. The `UNIQUE` constraint in the schema enforces this. If a job re-runs for the same data window, the `INSERT ... ON CONFLICT DO NOTHING` is a no-op.

```python
def upsert_signal(sig: Signal, conn):
    conn.execute("""
        INSERT INTO signals (run_id, sector_id, type, value, threshold,
                             window_start, window_end, context, severity)
        VALUES (...)
        ON CONFLICT (run_id, sector_id, type, window_start, window_end)
        DO UPDATE SET value = EXCLUDED.value, context = EXCLUDED.context
    """)
```

### Level 3: Digest is a pure function of its inputs

The `build_digest` job takes the current state of `signals` and `sector_aggregates` and produces a **new** digest row. It does not append to a previous digest. Each run produces exactly one digest. If the run is re-executed, the new digest supersedes the old one (or you can version them).

```python
def build_digest(run_id: str, signals: list, aggregates: dict) -> Digest:
    sections = []
    for sector in SECTORS:
        sector_signals = [s for s in signals if s.sector_id == sector.id]
        if sector_signals:
            section = build_section(sector, sector_signals, aggregates)
            sections.append(section)
    return Digest(run_id=run_id, sections=sections, generated_at=now())
```

**Re-run guarantee**: Running the pipeline twice for the same day with the same inputs produces an identical digest (modulo timestamps). This is critical for debugging and for "what would the digest have looked like yesterday if we'd had the 15:00 data?"

### Idempotency Key for the Run Itself

Each run has a `run_id` derived from the **scheduled window**, not from wall-clock time:

```python
run_id = f"2025-01-15_06:00"   # the cron slot, not the actual start time
```

If the scheduler crashes and restarts at 06:03, it sees that a run for `2025-01-15_06:00` already exists and either resumes it or (if it completed) does nothing. This prevents double-delivery.

---

## 4. Scheduling

### Why not Airflow/Prefect/Dagster?

For ~8–12 jobs on a single machine (or a small pair), a full orchestration framework is overkill. The complexity you introduce (state management, UI, plugin system) exceeds the problem. A 200-line scheduler in Python is sufficient and gives you full control over failure semantics.

That said, if you outgrow it, the job manifest is directly translatable to a Prefect `@flow` / `@task` DAG with `retries` and `trigger` semantics.

### Scheduler Implementation Sketch

```python
class Scheduler:
    def __init__(self, manifest: JobManifest, db: Connection):
        self.manifest = manifest
        self.db = db
        self.cron = CronParser(manifest.schedule)

    def tick(self):
        """Called every 30s by the main loop."""
        now = utcnow()
        if not self.cron.should_fire(now):
            return
        run_id = self.cron.current_slot_id(now)
        if self.db.run_exists(run_id, status="success"):
            return  # already ran
        self.execute_run(run_id)

    def execute_run(self, run_id: str):
        # Topologically sort
        order = topo_sort(self.manifest.jobs)
        results: dict[str, JobResult] = {}

        for job in order:
            deps = [j for j in job.depends_on]
            # Check if all non-optional deps succeeded
            unsatisfied = [
                d for d in deps
                if results[d].status == "failed"
                and self.is_output_required(job, d)
            ]
            if unsatisfied:
                results[job.id] = JobResult(status="skipped",
                                            reason=f"dep failed: {unsatisfied}")
                continue

            result = self.spawn_and_wait(job, run_id, results)
            results[job.id] = result

        self.db.finalize_run(run_id, results)
```

### Parallelism

Independent jobs (the three ingest jobs) run in parallel via `concurrent.futures.ProcessPoolExecutor` or simply by spawning them and using `asyncio.wait`. The scheduler uses a semaphore to cap concurrency if needed.

### Catch-up Logic

If the machine was off at 06:00 and comes back at 08:00, the scheduler checks: "Is there a cron slot that fired but has no completed run?" If yes, it executes it. This is the "catch-up" path. It uses the same `run_id` derivation, so it's idempotent.

---

## 5. The LLM Brain: Safe Integration

This is the most delicate part of the system. The LLM is **not** a reasoner here. It is a **narrative renderer** for a pre-computed structure.

### Architecture

```
┌────────────────────────────────────────────────────────────────┐
│  render_narrative job                                          │
│                                                                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  INPUT: digest_structure (JSON)                          │  │
│  │  - Contains ONLY Claim objects with provenance           │  │
│  │  - No raw data, no raw news text, no raw prices         │  │
│  │  - Every fact is already a sentence with a ref           │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  LLM CALL                                                │  │
│  │  - Model: (e.g., claude-sonnet, gpt-4o)                 │  │
│  │  - Prompt: strict system prompt (see below)             │  │
│  │  - Output: narrative text                               │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  VALIDATOR (deterministic, runs locally)                 │  │
│  │  1. Extract all numeric tokens from LLM output           │  │
│  │  2. Verify each appears in the input digest_structure   │  │
│  │  3. Extract all entity mentions (tickers, dates)        │  │
│  │  4. Verify each maps to a claim in the input            │  │
│  │  5. Check for "hedging language" that implies            │  │
│  │     uncertainty not present in the source               │  │
│  │  6. If any check fails → strip the offending sentence   │  │
│  │     or mark the narrative as "unvalidated"              │  │
│  └──────────────────────┬───────────────────────────────────┘  │
│                         │                                      │
│                         ▼                                      │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  OUTPUT: narrative_text (validated) + validation_report  │  │
│  └──────────────────────────────────────────────────────────┘  │
└────────────────────────────────────────────────────────────────┘
```

### The System Prompt

```text
You are a financial news narrative renderer. You receive a structured
digest of pre-verified claims. Your ONLY task is to write a flowing
morning brief that conveys the same information in prose.

RULES:
1. You may ONLY reference facts that appear in the input.
2. You may NOT introduce any number, date, ticker, or entity that
   is not in the input.
3. You may rephrase, reorder, and combine sentences.
4. You may add transitional language ("Meanwhile," "In contrast,").
5. You must preserve the severity level of each claim.
6. If a claim has low confidence, you must reflect that in your
   language (e.g., "reported to be," "per [source]").
7. Do NOT speculate. Do NOT extrapolate. Do NOT say "this could
   lead to" or "analysts expect."
8. Every factual assertion must be traceable to a claim ID.
   Append a [C:claim_id] tag after each factual sentence.

INPUT FORMAT:
- claims: [{id, text, provenance, confidence}]
- sections: [{title, claims: [ids]}]

OUTPUT: A single narrative, 400-800 words, with [C:...] tags.
```

### The Validator (Deterministic, Local)

```python
class NarrativeValidator:
    def __init__(self, digest_structure: DigestStructure):
        self.valid_numbers = self._extract_numbers(digest_structure)
        self.valid_entities = self._extract_entities(digest_structure)
        self.valid_claim_texts = {c.text for c in digest_structure.all_claims()}
        self.claim_id_to_text = {c.id: c.text for c in digest_structure.all_claims()}

    def validate(self, narrative: str) -> ValidationReport:
        issues = []
        sentences = split_sentences(narrative)
        
        for sent in sentences:
            # Check: every number in this sentence must be in valid_numbers
            for num in extract_numbers(sent):
                if num not in self.valid_numbers:
                    issues.append(UnvalidatedNumber(sentence=sent, number=num))
            
            # Check: every ticker/entity must be in valid_entities
            for ent in extract_entities(sent):
                if ent not in self.valid_entities:
                    issues.append(UnvalidatedEntity(sentence=sent, entity=ent))
            
            # Check: the sentence must map to at least one claim
            claim_ids = extract_claim_tags(sent)
            if not claim_ids:
                issues.append(MissingProvenance(sentence=sent))
            for cid in claim_ids:
                if cid not in self.claim_id_to_text:
                    issues.append(InvalidClaimRef(sentence=sent, ref=cid))
        
        return ValidationReport(
            passed=len(issues) == 0,
            issues=issues,
            cleaned_text=self._strip_offending(sentences, issues)
        )
```

**Key design choice**: The validator does NOT try to verify semantic entailment (that's what the LLM is for). It verifies **referential integrity** — every concrete reference (number, entity, date) in the output must exist in the input. This catches hallucinated numbers, phantom tickers, and fabricated sources without requiring a second LLM call.

### What the LLM Can and Cannot Do

| Capability | Allowed? | Enforcement |
|-----------|----------|-------------|
| Rephrase a claim | ✅ | Semantic — human review or spot-check |
| Combine two claims into one sentence | ✅ | Each must carry its claim tag |
| Add a transitional phrase | ✅ | Contains no factual content |
| Introduce a new number | ❌ | Validator: number not in input → strip |
| Mention a ticker not in the digest | ❌ | Validator: entity not in input → strip |
| Add speculative language | ❌ | Prompt rule + validator flag (optional NLI check) |
| Change the severity of a claim | ❌ | Prompt rule (hard to enforce automatically; rely on prompt + human review) |

### Optional: NLI Spot-Check

For high-stakes claims (severity `alert` or `critical`), you can run a small NLI model (e.g., DeBERTa-v3-mnli) to check that the narrative sentence **entails** the original claim. This is expensive and slow, so it's opt-in and only for the top-N most important claims.

---

## 6. The Digest: Structure and Auditability

### Digest Structure (what the LLM sees)

```json
{
  "run_id": "2025-01-15_06:00",
  "generated_at": "2025-01-15T06:14:22Z",
  "data_gaps": [
    {"source": "ingest_news", "reason": "API timeout after 3 retries", "impact": "may be missing 2-3 stories from Reuters"}
  ],
  "sections": [
    {
      "title": "Sector Signals: AI Infrastructure",
      "claims": [
        {
          "id": "c001",
          "text": "NVIDIA's 20-day VWAP is 4.2% above its 50-day VWAP, exceeding the 3% tripwire threshold.",
          "provenance": {
            "kind": "computed",
            "ref_type": "signal",
            "ref_id": "sig_20250115_nvda_vwap_cross",
            "computation": "vwap_cross_detector(ticker=NVDA, short=20, long=50, threshold=0.03)"
          },
          "confidence": {"level": "exact", "detail": "Computed from closing prices 2025-01-15 and prior 50 sessions."}
        },
        {
          "id": "c002",
          "text": "NVIDIA reported Q3 revenue of $35.1B, beating consensus by 8.3%. Source: NVIDIA IR, 2025-01-28.",
          "provenance": {
            "kind": "source",
            "ref_type": "earnings",
            "ref_id": "ern_nvda_2025q3"
          },
          "confidence": {"level": "reported", "detail": "Company-reported figure from official press release."}
        }
      ]
    },
    {
      "title": "News Flow",
      "claims": [
        {
          "id": "c003",
          "text": "Reuters reported (2025-01-14) that three hyperscaler customers expanded their H100 purchase agreements by an undisclosed amount.",
          "provenance": {
            "kind": "source",
            "ref_type": "news",
            "ref_id": "news_reuters_20250114_h100"
          },
          "confidence": {"level": "reported", "detail": "Single-source report; no independent confirmation in ingested feed."}
        }
      ]
    }
  ]
}
```

### Audit Trail

Given any sentence in the final narrative, you can trace:

1. **Narrative sentence** → claim tag `[C:c001]`
2. **Claim c001** → provenance: `computed`, function `vwap_cross_detector`, inputs hash `sha256:abc...`
3. **Inputs** → exact `market_data` rows (ticker=NVIDIA, dates=...)
4. **Raw source** → the original price feed entry with timestamp and source attribution

This is a complete, deterministic audit chain from prose back to raw data.

### Immutability

Digests are **append-only**. Once written, they are never modified. The `narrative_text` field is filled in by the `render_narrative` job and is part of the digest row's final state. If you want to re-render (e.g., after a model upgrade), you create a new row with a `parent_digest_id` pointer.

---

## 7. Key Trade-offs

### 1. Determinism vs. Intelligence

| Choice | Pro | Con |
|--------|-----|-----|
| All-deterministic signals | Fully reproducible, auditable, fast | Misses nuance (e.g., "the tone of the earnings call shifted") |
| LLM analyzes raw news | Catches subtleties | Introduces non-determinism, hallucination risk, cost |

**Resolution**: Keep the signal layer fully deterministic. Use the LLM only for narrative. If you want the LLM to *contribute* to signal generation (e.g., "classify this news as a supply-chain risk"), do it in a **separate, sandboxed job** whose output is a structured classification that feeds into the deterministic signal layer. The LLM's output is then validated against a schema and can be overridden.

### 2. Latency vs. Completeness

By 06:00 EST, some data may not be available:
- Earnings reported after 04:00 ET may not be in the feed yet.
- News from AP/Reuters may have a 5–15 min ingestion lag.

**Resolution**: The `data_gaps` field in the digest explicitly states what's missing and why. The reader knows the digest is a *snapshot*, not a guarantee of completeness. You can also add a "supplemental update" at 07:00 that picks up late-arriving data.

### 3. Granularity: Sector vs. Ticker

12 sectors × 5–15 tickers each = 60–180 tickers. Full per-ticker signal computation is feasible but produces a long digest.

**Resolution**: Compute signals at the ticker level, but the digest is organized by sector. A sector section aggregates its tickers' signals, highlighting only those that crossed a threshold. Per-ticker detail is available in the raw data for drill-down.

### 4. LLM Validation Strictness

| Strictness | Effect |
|-----------|--------|
| Strip any unvalidated number | Very safe, but may produce awkward prose if the LLM "rounds" a number |
| Allow numbers within 0.1% of a valid number | Handles rounding, but requires a tolerance parameter |
| Only flag numbers that are wildly off (>1%) | Risky — a hallucinated "23%" vs. actual "3%" would pass |

**Resolution**: Use exact matching for all numbers. The LLM is instructed to copy numbers verbatim. If it rounds, the validator strips the sentence. In practice, modern LLMs follow "copy verbatim" instructions well when the numbers are in the prompt.

### 5. Single-Node vs. Distributed

For 12 sectors and ~100 tickers, the entire pipeline runs in <2 minutes on a single mid-range machine. Distributed computing (Kafka, Spark, Kubernetes) adds operational complexity for no benefit at this scale.

**Resolution**: Single process, single Postgres, single filesystem. If you scale to 50+ sectors or add real-time (sub-hour) signals, revisit.

### 6. Storage Retention

| Data | Retention | Rationale |
|------|-----------|-----------|
| Raw market data | 2 years | Needed for signal recomputation, backtesting |
| News items | 1 year | Provenance for digests from the last year |
| Digests | 5
