# mlx4 — q3 `q3_architecture` (architecture)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=469.3s decode=17.5 tok/s prefill=405.1 tok/s gen_tokens=8192 prompt_tokens=224 finish=length
- thinking chars: 5579

## Prompt

Design the architecture of an overnight research pipeline for an AI-infrastructure investing workstation. It must: ingest prices, news and earnings for ~12 sectors; evaluate deterministic tripwire risk signals; synthesize a ranked, evidence-backed morning digest where every insight traces to a computed number or a dated source; optionally have an LLM narrate (never fabricate) the already-true digest; and run on a cron scheduler. Cover the data model, the job registry + failure isolation (a failed step must never abort the chain), idempotency, scheduling, where the LLM brain plugs in safely, and how you keep the digest accurate and auditable. Discuss the key trade-offs and failure modes.

## Answer

# Overnight AI-Infrastructure Research Pipeline

## Design Philosophy

The system has one invariant: **the digest is a deterministic function of (market state, news state, earnings state, config)**. The LLM is a post-hoc narrator that maps a verified artifact into natural language. If the LLM step fails or produces unvalidated output, the template-rendered digest ships unchanged. No human judgment is required for correctness; the LLM adds only readability.

---

## 1. Data Model

### Core Entities

```sql
-- Universe definition (changes infrequently, versioned)
CREATE TABLE sector (
    id          UUID PRIMARY KEY,
    name        TEXT NOT NULL,          -- e.g. 'AI-HW'
    tickers     JSONB NOT NULL,        -- ["NVDA","AMD","AVGO",...]
    config      JSONB NOT NULL,        -- thresholds, weights, per-ticker overrides
    version     INT NOT NULL,
    effective_from TIMESTAMPTZ NOT NULL,
    UNIQUE(name, effective_from)
);

CREATE TABLE ticker (
    id          UUID PRIMARY KEY,
    symbol      TEXT NOT NULL UNIQUE,
    sector_id   UUID REFERENCES sector(id),
    name        TEXT NOT NULL
);

-- Observed market data (append-only, immutable)
CREATE TABLE price_obs (
    ticker_id   UUID REFERENCES ticker(id),
    ts          TIMESTAMPTZ NOT NULL,   -- exchange time
    open        NUMERIC(12,4),
    high        NUMERIC(12,4),
    low         NUMERIC(12,4),
    close       NUMERIC(12,4),
    volume      BIGINT,
    source      TEXT NOT NULL,           -- 'polygon','alphavantage',...
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker_id, ts, source)
);

-- News events (deduplicated by content hash)
CREATE TABLE news_item (
    id          UUID PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL,   -- publication time
    source      TEXT NOT NULL,
    url         TEXT NOT NULL,
    headline    TEXT NOT NULL,
    body_hash   TEXT NOT NULL,           -- SHA256 of canonical body
    entities    JSONB NOT NULL,         -- ["NVDA","data-center",...]
    sector_id   UUID REFERENCES sector(id),
    relevance   NUMERIC(4,3),           -- 0..1, computed by classifier
    UNIQUE(source, url, ts)
);

-- Earnings events (one per ticker per fiscal quarter)
CREATE TABLE earnings_event (
    ticker_id   UUID REFERENCES ticker(id),
    fiscal_q    TEXT NOT NULL,           -- '2025Q4'
    ts          TIMESTAMPTZ NOT NULL,   -- announcement time
    reported_eps   NUMERIC(10,4),
    expected_eps   NUMERIC(10,4),
    reported_rev   NUMERIC(16,2),
    expected_rev   NUMERIC(16,2),
    guidance     JSONB,                 -- next-quarter guidance if provided
    source      TEXT NOT NULL,
    ingested_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (ticker_id, fiscal_q)
);

-- Computed signals (the core of the analysis)
CREATE TABLE signal (
    id          UUID PRIMARY KEY,
    ts          TIMESTAMPTZ NOT NULL,   -- computed-at time
    type        TEXT NOT NULL,           -- 'drawdown','volume_spike','earnings_miss',...
    severity    SMALLINT NOT NULL,       -- 1=watch, 2=alert, 3=critical
    ticker_id   UUID REFERENCES ticker(id),
    sector_id   UUID REFERENCES sector(id),
    trigger_val NUMERIC(12,6) NOT NULL,  -- the actual computed value
    threshold   NUMERIC(12,6) NOT NULL,  -- the config threshold that was crossed
    formula     TEXT NOT NULL,           -- human-readable: "close < sma_20 * (1 - 0.05)"
    formula_hash TEXT NOT NULL,          -- SHA256 of formula + config version
    inputs_snapshot JSONB NOT NULL,      -- the exact inputs used (for audit/replay)
    job_run_id  UUID REFERENCES job_run(id)
);

-- The output: ranked digest entries
CREATE TABLE digest_entry (
    id          UUID PRIMARY KEY,
    run_id      UUID REFERENCES digest_run(id),
    rank        INT NOT NULL,
    signal_id   UUID REFERENCES signal(id),
    title       TEXT NOT NULL,
    body        TEXT NOT NULL,           -- template-rendered (always true)
    evidence    JSONB NOT NULL,          -- [{type:'price',ref:'NVDA 2025-01-15 close=128.44'},...]
    provenance  TEXT NOT NULL,           -- SHA256(canonical_json(signal.inputs_snapshot))
    confidence  NUMERIC(3,2) NOT NULL    -- 0..1, based on input completeness
);

CREATE TABLE digest_run (
    id          UUID PRIMARY KEY,
    generated_at TIMESTAMPTZ NOT NULL,
    code_version TEXT NOT NULL,          -- git SHA
    config_hash TEXT NOT NULL,           -- SHA256 of all sector configs
    data_window JSONB NOT NULL,          -- {from: ts, to: ts}
    completeness JSONB NOT NULL,         -- {sector: 0.87, ...} per-sector data availability
    llm_narrated BOOLEAN NOT NULL DEFAULT false,
    llm_model   TEXT,
    entry_count INT NOT NULL
);
```

### The Critical Design Decision: `inputs_snapshot`

Every signal stores the *exact* inputs it was computed from. This is the backbone of auditability. If someone asks "why did you flag NVDA at 03:12 UTC?", you can replay the computation from the snapshot and reproduce the result bit-for-bit. This also makes backtesting and config changes testable: you can recompute all signals for a historical window with a new threshold and diff the results.

---

## 2. Job Registry and Failure Isolation

### Job DAG

The pipeline is a DAG of jobs, not a monolithic script. Each node is independently executable, independently retriable, and independently failure-tolerant.

```yaml
# jobs.yaml (the job graph definition)

pipeline: overnight_research
window:
  start: "22:00"   # UTC, post-market-close
  end:   "06:00"   # UTC, before market open
  digest_deadline: "05:30"  # hard deadline; if missed, ship what you have

jobs:
  # ── Layer 0: Ingestion (parallel, one job per source×entity) ──
  - id: ingest_prices:{sector}
    type: stream_batch
    inputs: [sector]
    output: price_obs rows
    timeout: 300s
    retry: {max: 2, backoff: exponential}
    idempotency_key: "ingest_prices:{sector}:{window_start}:{window_end}"

  - id: ingest_news:{sector}
    type: batch
    inputs: [sector]
    output: news_item rows
    timeout: 600s
    retry: {max: 3, backoff: exponential}
    idempotency_key: "ingest_news:{sector}:{window_start}"

  - id: ingest_earnings:{sector}
    type: batch
    inputs: [sector]
    output: earnings_event rows
    timeout: 300s
    retry: {max: 1}  # earnings are rare; don't hammer
    idempotency_key: "ingest_earnings:{sector}:{fiscal_quarter}"

  # ── Layer 1: Signal Computation (parallel per sector) ──
  - id: compute_signals:{sector}
    type: transform
    depends_on: [ingest_prices:{sector}, ingest_news:{sector}, ingest_earnings:{sector}]
    inputs: [price_obs, news_item, earnings_event, sector_config]
    output: signal rows
    timeout: 120s
    retry: {max: 1}  # deterministic; retry is only for I/O errors
    idempotency_key: "compute_signals:{sector}:{code_version}:{config_hash}:{data_window}"

  # ── Layer 2: Cross-sector aggregation ──
  - id: aggregate_sector_risk
    type: transform
    depends_on: ["compute_signals:*"]   # wildcard: all sectors
    inputs: [signals from all sectors]
    output: sector_risk_scores
    timeout: 60s
    idempotency_key: "aggregate:{code_version}:{config_hash}:{data_window}"

  # ── Layer 3: Digest Synthesis ──
  - id: synthesize_digest
    type: transform
    depends_on: [aggregate_sector_risk, "compute_signals:*"]
    inputs: [all signals, sector_risk_scores, completeness_map]
    output: digest_entry rows + digest_run record
    timeout: 60s
    idempotency_key: "synthesize:{code_version}:{config_hash}:{data_window}"

  # ── Layer 4: LLM Narration (optional, best-effort) ──
  - id: narrate_digest
    type: llm
    depends_on: [synthesize_digest]
    inputs: [digest entries, provenance data]
    output: narrated text (or fallback to template)
    timeout: 300s
    retry: {max: 1}
    failure_policy: degrade   # ← critical: failure here does NOT affect digest correctness
    idempotency_key: "narrate:{digest_run_id}:{llm_model}:{code_version}"

  # ── Layer 5: Delivery ──
  - id: deliver_digest
    type: emit
    depends_on: [synthesize_digest, narrate_digest]  # uses narrate if available, else falls back
    inputs: [digest, narration (optional)]
    output: email / Slack / dashboard
    timeout: 30s
    failure_policy: retry_then_alert
```

### Failure Isolation: The "Partial Truth" Contract

The critical insight is that **the digest is designed to be complete with partial input**. The `synthesize_digest` job receives a `completeness_map`:

```python
# Inside synthesize_digest
def build_digest(signals, risk_scores, completeness_map):
    """
    completeness_map: {sector_id: float} where 1.0 = all expected data present.
    A sector at 0.0 produces no signals but IS reported in the digest
    with an explicit "DATA UNAVAILABLE" entry.
    """
    entries = []
    for sector, completeness in completeness_map.items():
        if completeness < MIN_VIABLE_THRESHOLD:  # e.g. 0.3
            entries.append(
                DataGapEntry(
                    sector=sector,
                    completeness=completeness,
                    body=f"[{sector}] Data availability at {completeness:.0%}. "
                         f"Signals for this sector are unreliable; "
                         f"no risk assessment provided."
                )
            )
            continue
        # Normal path: rank signals for this sector
        sector_signals = [s for s in signals if s.sector_id == sector]
        entries.extend(rank_and_template(sector_signals, completeness))

    # Cross-sector entries (systemic signals)
    entries.extend(risk_scores.to_entries())

    return rank_all(entries)  # global ranking
```

**The rule: a failed ingestion job reduces completeness for its sector. It never blocks downstream jobs. The digest always ships, always reflects what it knows, and always flags what it doesn't.**

### Job Run Tracking

```sql
CREATE TABLE job_run (
    id            UUID PRIMARY KEY,
    job_id        TEXT NOT NULL,          -- "ingest_prices:AI-HW"
    pipeline      TEXT NOT NULL,
    scheduled_at  TIMESTAMPTZ NOT NULL,
    started_at    TIMESTAMPTZ,
    finished_at   TIMESTAMPTZ,
    status        TEXT NOT NULL,          -- 'pending','running','success','failed','skipped','degraded'
    error         JSONB,                  -- {type, message, traceback_hash}
    inputs_hash   TEXT,                   -- SHA256 of canonical input snapshot
    outputs_hash  TEXT,                   -- SHA256 of canonical output snapshot
    code_version  TEXT NOT NULL,          -- git SHA at execution time
    parent_run    UUID,                   -- for retry chain
    UNIQUE(job_id, scheduled_at, code_version)
);
```

The `status = 'degraded'` state is key: the job ran, produced output, but the output is known-incomplete (e.g., price feed was available but news feed wasn't, so signals are missing the news component). Downstream jobs receive the degraded output and factor it into their completeness calculation.

---

## 3. Idempotency

### The Three Idempotency Layers

| Layer | Key | Guarantee |
|-------|-----|-----------|
| **Input idempotency** | `(source, entity, timestamp)` for prices; `(source, url, ts)` for news; `(ticker, fiscal_q)` for earnings | Re-ingesting the same data does not create duplicates. We use `ON CONFLICT DO NOTHING`. |
| **Computation idempotency** | `inputs_hash = SHA256(canonical_json(all_relevant_input_rows + config + code_version))` | If the job has already run with this exact input hash, skip. Store the hash in `job_run.inputs_hash`. |
| **Output idempotency** | `idempotency_key` (the YAML string, rendered) | If a job with this key already produced a `success` or `degraded` run, the orchestrator skips re-dispatch. |

### Implementation

```python
class IdempotentJob(ABC):
    def execute(self, ctx: JobContext) -> JobResult:
        # 1. Compute the canonical input hash
        input_hash = sha256(canonical_json(self.gather_inputs(ctx)))

        # 2. Check if we've already done this exact work
        existing = self.db.query(
            "SELECT outputs_hash FROM job_run "
            "WHERE job_id = %s AND inputs_hash = %s AND status IN ('success','degraded')",
            (self.id, input_hash)
        )
        if existing:
            return JobResult(status='skipped', reason='idempotent_hit',
                             outputs_hash=existing[0])

        # 3. Execute
        try:
            outputs = self._do_work(ctx)
            outputs_hash = sha256(canonical_json(outputs))
            self.db.insert_job_run(
                id=self.id, inputs_hash=input_hash,
                outputs_hash=outputs_hash, status=self._assess_status(ctx)
            )
            return JobResult(status=self._assess_status(ctx), outputs_hash=outputs_hash)
        except Exception as e:
            self.db.insert_job_run(
                id=self.id, inputs_hash=input_hash,
                status='failed', error=serialize_error(e)
            )
            raise
```

### Why `code_version` Is in the Hash

If you change the signal formula (e.g., adjust a moving-average window from 20 to 25), the inputs are the same but the correct output is different. Including the git SHA in the computation hash means a config/code change automatically invalidates all prior results. This is what makes "recompute with new config" safe: the old results remain (for audit) but new results get new hashes.

---

## 4. Scheduling

### Cron Entry Point

```cron
# /etc/cron.d/research-pipeline (or systemd timer)
# Runs at 22:00 UTC every trading day
0 22 * * 1-5 /opt/research/bin/run_pipeline.sh
```

```bash
#!/usr/bin/env bash
# run_pipeline.sh — thin wrapper, all logic in Python
set -euo pipefail

PIPELINE="overnight_research"
WINDOW_START="${PIPELINE_START:-$(date -u -d '0 22 hours' +%Y-%m-%dT%H:%M:%SZ)}"
WINDOW_END="${PIPELINE_END:-$(date -u -d '0 23 hours' +%Y-%m-%dT%H:%M:%SZ)}"

python -m research.orchestrator \
    --pipeline "$PIPELINE" \
    --window-start "$WINDOW_START" \
    --window-end "$WINDOW_END" \
    --config /opt/research/config/ \
    --code-version "$(git -C /opt/research rev-parse HEAD)"
```

### The Orchestrator (Not Airflow, Not Prefect — Just Python)

For a 12-sector, 5-layer pipeline with ~50 total jobs, a full orchestrator is overkill and adds its own failure modes. A lightweight DAG runner suffices:

```python
class DAGRunner:
    """
    Executes a job graph in topological order.
    - Jobs with the same layer run in parallel (ThreadPoolExecutor).
    - A failed job marks its downstream dependents as 'skipped' with reason.
    - The digest job ALWAYS runs; it adapts to whatever is available.
    """

    def __init__(self, graph: JobGraph, db: DB, max_parallel: int = 12):
        self.graph = graph
        self.db = db
        self.results: dict[str, JobResult] = {}

    def run(self):
        for layer in self.graph.layers():  # topological layers
            runnable = []
            for job in layer:
                if self._should_skip(job):
                    self.results[job.id] = JobResult(
                        status='skipped',
                        reason=f"upstream {job.depends_on[0]} did not succeed"
                    )
                    self.db.record_run(job.id, self.results[job.id])
                    continue
                runnable.append(job)

            with ThreadPoolExecutor(max_workers=self.max_parallel) as pool:
                futures = {pool.submit(job.execute, self.ctx): job for job in runnable}
                for fut in as_completed(futures):
                    job = futures[fut]
                    try:
                        self.results[job.id] = fut.result()
                    except Exception as e:
                        self.results[job.id] = JobResult(status='failed', error=str(e))
                    self.db.record_run(job.id, self.results[job.id])

        # The digest is the terminal node; it always gets called
        digest_job = self.graph.terminal()
        self.results[digest_job.id] = digest_job.execute(self.ctx)
        self.db.record_run(digest_job.id, self.results[digest_job.id])
```

### Why Not Airflow?

| Concern | Airflow | Custom DAG Runner |
|---------|---------|-------------------|
| Operational complexity | High (Postgres, scheduler, webserver, workers) | Low (one process, one DB connection) |
| Failure semantics | Task retries, but the whole DAG can be in a broken state | Explicit per-job status; orchestrator never gets "stuck" |
| Idempotency | You still have to implement it | Built into the job base class |
| Partial execution | Requires custom logic to "skip but continue" | Native: the runner just checks dependents |
| Scale | Overkill for ~50 jobs/night | Perfect fit |

The only reason to use Airflow would be if you later want to scale to hundreds of sectors or add complex cross-pipeline dependencies. For 12 sectors, a 200-line Python orchestrator is the right tool.

---

## 5. The LLM Brain: Safe Integration

### The Safety Contract

The LLM operates under a strict contract with three rules:

1. **Input is the ground truth.** The LLM receives the rendered digest (template-produced, factually verified) and a provenance manifest. It may *rephrase, reorder, summarize, or add transitional language*. It may not *assert new facts, add numbers, change severity, or introduce entities not in the input*.

2. **Output is validated before use.** A validation step checks the LLM output against the input. Any claim in the output that doesn't map to a claim in the input is a violation.

3. **Failure means fallback.** If validation fails, or the LLM call times out, or the model is unavailable, the template-rendered digest ships as-is. The user never sees a wrong number.

### Prompt Structure

```text
SYSTEM:
You are a financial research narrator. You will be given a structured digest
of market signals. Your job is to rewrite it as a readable morning brief.

RULES (absolute, no exceptions):
- You may ONLY state facts that appear in the INPUT.
- You may NOT introduce any ticker, number, date, or entity not in the INPUT.
- You may NOT change any number, threshold, or severity level.
- You may rephrase, reorder, group, and add transitional language.
- You may say "no material signals for [sector]" if the input says so.
- If a sector is marked "DATA UNAVAILABLE", you must preserve that status.
- Output in the format: a single markdown document with ## headers per sector.

INPUT:
{structured_digest_json}

PROVENANCE MANIFEST:
{evidence_table}
```

### Validation

```python
class LLMValidator:
    """
    Post-LLM validation. Extracts all "factual claims" from the LLM output
    and checks each one exists in the input digest.
    """

    def validate(self, llm_output: str, input_digest: list[DigestEntry]) -> ValidationReport:
        claims = self._extract_claims(llm_output)
        input_facts = self._extract_facts(input_digest)

        violations = []
        for claim in claims:
            if not self._matches_any(claim, input_facts):
                violations.append(claim)

        return ValidationReport(
            passed=len(violations) == 0,
            violations=violations,
            total_claims=len(claims)
        )

    def _extract_claims(self, text: str) -> list[FactClaim]:
        """
        Extracts: all numbers, all tickers, all dates, all named entities
        from the LLM output. Uses regex + NER for tickers/dates.
        """
        ...

    def _matches_any(self, claim: FactClaim, facts: set[FactClaim]) -> bool:
        """
        Numeric claims must match exactly (within floating-point tolerance).
        Entity claims must match the entity in the input.
        Date claims must match the timestamp in the input.
        """
        ...
```

### The "Never Fabricate" Guarantee, Formalized

```
Theorem: If the LLM output passes validation, every factual claim in the
output is entailed by the input digest.

Proof sketch: Validation extracts all atomic facts (numbers, entities, dates)
from the output and requires each to have a matching fact in the input.
Since the input digest was produced by deterministic computation from raw data,
and each signal in the input has a provenance_hash back to its inputs,
the transitive chain is:
    LLM output claim → input digest fact → signal.inputs_snapshot → raw data
∎
```

The practical risk is not in the proof but in the *extraction* step. If `_extract_claims` misses a claim (e.g., the LLM says "the stock fell sharply" where "sharply" implies a degree not in the input), validation passes but the claim is subtly ungrounded. Mitigation:

- The validation step also checks for *degree words* and *hedges* that imply information not in the input.
- We log all LLM inputs/outputs and sample 10% for human review weekly.
- The template-rendered digest is always available as the "canonical" version; the LLM version is explicitly labeled "AI-narrated summary of the following verified data."

---

## 6. Accuracy and Auditability

### The Provenance Chain

Every claim in the final deliverable traces through this chain:

```
[Final digest text]
    │
    │  (LLM narration OR template)
    ▼
[Digest entry: "NVDA down 7.2% below 20-day SMA, volume 2.3× average"]
    │
    │  provenance_hash = SHA256(inputs_snapshot)
    ▼
[Signal row: type=drawdown, trigger_val=0.928, threshold=0.95]
    │
    │  inputs_snapshot = {close: 128.44, sma_20: 138.41, vol_20d_avg: 42M, vol_today: 97M}
    ▼
[Raw data: price_obs(NVDA, 2025-01-15, close=128.44, source=polygon)]
              price_obs(NVDA, 2025-01-01..14, close=[...])  ← 20 observations
    │
    ▼
[External source: Polygon.io API, timestamp 2025-01-15T21:12:00Z]
```

To audit: given any line in the digest, a human or script can follow the chain to the raw data and verify the computation. The `inputs_snapshot` in the signal table makes this a single DB query.

### Reproducibility

The `digest_run` table stores:
- `code_version`: exact git SHA
- `config_hash`: SHA256 of all sector configs at run time
- `data_window`: exact time range

Given these three values, the entire pipeline can be re-run and must produce bit-identical output (barring data source corrections). This is the "reproducibility key" — include it in every digest so a reader can request a replay.

---

## 7. Key Trade-offs

### 7.1: Staleness vs. Completeness

**The dilemma:** The pipeline starts at 22:00 UTC. If the news API is slow, do you wait for it or proceed without it?

**Resolution:** Per-source deadlines. Each ingestion job has a `deadline` (e.g., news must arrive by 00:00 UTC). If the deadline passes without data, the job records `status='degraded'` and the pipeline continues. The digest explicitly states which sources were unavailable.

The trade-off: you trade *recency* (you might miss a 23:58 news item that arrives at 00:02) for *availability* (the digest always ships on time). For an investing workstation, availability wins: a digest that says "no news data" is always better than a digest that's 4 hours late.

### 7.2: Signal Granularity vs. Alert Fatigue

**The dilemma:** With 12 sectors × ~15 tickers each × ~8 signal types, you could generate hundreds of signals per night. The user will ignore a 200-item digest.

**Resolution:** A two-tier ranking:
- **Tier 1 (Critical):** `severity=3` signals. Always included, ranked by magnitude of threshold breach.
- **Tier 2 (Watch):** `severity=2` signals. Included only if the sector's aggregate risk score exceeds a threshold. Otherwise collapsed into a one-line sector summary.
- **Tier 3 (Noise):** `severity=1` signals. Aggregated into a single "background" section with counts only.

The user configures the ratio. The trade-off is between sensitivity (catching early warnings) and signal-to-noise ratio. For an overnight pipeline, I'd start with a bias toward fewer, higher-confidence signals and tune up over time.

### 7.3: Deterministic Signals vs. LLM "Insight"

**The dilemma:** Purely deterministic signals (threshold crossings, ratio changes) are auditable but may miss emergent patterns. An LLM could "see" that three different news items together imply a regulatory risk. But if the LLM identifies a pattern, how do you audit it?

**Resolution:** The LLM does not identify new signals. It can only *narrate* existing ones. If you want the LLM to propose new signals, that's a separate pipeline (a "hypothesis generation" step) whose output goes to a human for review, not into the automated digest. The automated digest remains 100% deterministic.

This is the most important trade-off in the system: **you sacrifice potential insight for guaranteed correctness.** In my view, for an investing workstation where a wrong number could lead to a bad trade, this is the right trade. The LLM's value is in *readability and synthesis* (grouping related signals, explaining *why* a cluster of signals matters), not in *discovery*.

### 7.4: Idempotency vs. Data Corrections

**The dilemma:** If Polygon later corrects a price (e.g., a bad tick was published and then retracted), the idempotency key says "already computed, skip." But the data has changed.

**Resolution:** Distinguish between *initial ingestion* (idempotent by content) and *corrections* (a separate `ingest_corrections` job that watches for retracted/updated data). Corrections trigger a *recomputation* with a new `inputs_hash` (because the input rows changed), producing a *new* signal row alongside the old one (which is marked `superseded=true`). The digest uses the latest non-superseded signals.

This preserves the audit trail: you can see the original signal, the correction, and the recomputed signal, all in the same table.

### 7.5: Sector Independence vs. Systemic Risk

**The dilemma:** The jobs are per-sector. But a macro event (Fed announcement, geopolitical shock) hits all 12 sectors simultaneously. Treating it as 12 independent "failures" or 12 independent "signals" is wrong — it's one event with 12 manifestations.

**Resolution:** A `cross_sector_signal` layer (the `aggregate_sector_risk` job). It looks at the *correlation* of signals across sectors. If 8+ sectors have `severity ≥ 2` signals within the same 15-minute window, it emits a single `type=systemic` signal at `severity=3` that *supersedes* the individual sector signals in the digest (they're referenced as evidence but not listed individually).

---

##
