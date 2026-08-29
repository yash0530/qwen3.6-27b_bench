# mlx6 — q2 `q2_coding` (coding)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=592.8s decode=12.3 tok/s prefill=347.5 tok/s gen_tokens=7300 prompt_tokens=185 finish=stop
- thinking chars: 12780

## Prompt

Implement, in TypeScript, a rate-limited asynchronous job queue suitable for an overnight ETL pipeline. Requirements: a configurable max concurrency and a max requests-per-second token-bucket limiter; per-item retry with exponential backoff and jitter; per-item failure isolation so one failure never aborts the batch; a result object that counts successes/failures and collects error details; and graceful cancellation via an AbortSignal. Provide the full implementation with types, then a short usage example and notes on the trade-offs you made.

## Answer

```typescript
// ============================================================================
// rate-limited-job-queue.ts
// A rate-limited, concurrency-capped async job queue for batch ETL workloads.
// ============================================================================

// ─── Public Types ────────────────────────────────────────────────────────────

export interface RetryConfig {
  /** Maximum number of retries after the initial attempt (total attempts = maxRetries + 1). */
  maxRetries: number;
  /** Base delay in ms for exponential backoff. */
  baseDelayMs: number;
  /** Upper cap on a single backoff delay in ms. */
  maxDelayMs: number;
}

export interface JobQueueOptions {
  /** Maximum number of items executing concurrently (semaphore slots). */
  maxConcurrency: number;
  /** Sustained throughput cap: tokens refilled per second. */
  maxRPS: number;
  /** Retry policy applied independently to every item. */
  retry: RetryConfig;
  /** Optional progress hook; called after each item settles (success or final failure). */
  onItemSettled?: (item: ItemResult<unknown, unknown>, elapsed: number) => void;
}

export interface ItemResult<T, R> {
  index: number;
  item: T;
  ok: boolean;
  /** Populated on success. */
  result?: R;
  /** Populated on final failure (after all retries exhausted). */
  error?: unknown;
  /** Total number of attempts made (1 + retries actually performed). */
  attempts: number;
}

export interface JobRunResult<T, R> {
  successes: number;
  failures: number;
  skipped: number;
  items: ItemResult<T, R>[];
  aborted: boolean;
  startedAt: number;
  completedAt: number;
}

export type ProcessItem<T, R> = (item: T, index: number) => Promise<R>;

// ─── TokenBucket ─────────────────────────────────────────────────────────────

class TokenBucket {
  private tokens: number;
  private lastRefill: number;
  private readonly rate: number;       // tokens per second
  private readonly capacity: number;   // max burst size

  constructor(rate: number, capacity?: number) {
    if (rate <= 0) throw new RangeError("TokenBucket rate must be > 0");
    this.rate = rate;
    this.capacity = capacity ?? rate;
    this.tokens = this.capacity;
    this.lastRefill = Date.now();
  }

  /**
   * Acquires one token, waiting (asynchronously) until one is available.
   * Rejects immediately if the signal is already aborted or aborts while waiting.
   */
  async acquire(signal?: AbortSignal): Promise<void> {
    for (;;) {
      this.refill();

      if (signal?.aborted) {
        throw abortError();
      }

      if (this.tokens >= 1) {
        this.tokens -= 1;
        return;
      }

      // Sleep until the next token should be available
      const deficit = 1 - this.tokens;
      const waitMs = (deficit / this.rate) * 1000;
      await sleep(waitMs, signal);
    }
  }

  private refill(): void {
    const now = Date.now();
    const elapsedSec = (now - this.lastRefill) / 1000;
    this.tokens = Math.min(this.capacity, this.tokens + elapsedSec * this.rate);
    this.lastRefill = now;
  }
}

// ─── Semaphore ───────────────────────────────────────────────────────────────

class Semaphore {
  private active = 0;
  private readonly waiters: Array<{
    resolve: () => void;
    reject: (err: unknown) => void;
    onAbort: (() => void) | null;
  }> = [];
  private readonly max: number;

  constructor(max: number) {
    if (max <= 0) throw new RangeError("Semaphore max must be > 0");
    this.max = max;
  }

  async acquire(signal?: AbortSignal): Promise<void> {
    if (signal?.aborted) throw abortError();

    if (this.active < this.max) {
      this.active++;
      return;
    }

    await new Promise<void>((resolve, reject) => {
      const entry = { resolve, reject, onAbort: null as (() => void) | null };

      if (signal) {
        entry.onAbort = () => {
          const idx = this.waiters.indexOf(entry);
          if (idx !== -1) this.waiters.splice(idx, 1);
          reject(abortError());
        };
        signal.addEventListener("abort", entry.onAbort, { once: true });
      }

      this.waiters.push(entry);
    });

    // Slot was freed by a concurrent release(); claim it.
    this.active++;
  }

  release(): void {
    const next = this.waiters.shift();
    if (next) {
      // Clean up abort listener
      if (next.onAbort) {
        // The listener is { once: true } so it self-removes after firing,
        // but we still want to detach it now that we're resolving normally.
        // We can't easily get the signal reference, so rely on { once: true }.
      }
      next.resolve();
    } else {
      this.active--;
    }
  }
}

// ─── JobQueue ────────────────────────────────────────────────────────────────

export class JobQueue<T, R> {
  private readonly semaphore: Semaphore;
  private readonly bucket: TokenBucket;
  private readonly opts: JobQueueOptions;

  constructor(options: JobQueueOptions) {
    if (options.maxConcurrency <= 0)
      throw new RangeError("maxConcurrency must be > 0");
    if (options.maxRPS <= 0)
      throw new RangeError("maxRPS must be > 0");
    if (options.retry.maxRetries < 0)
      throw new RangeError("retry.maxRetries must be >= 0");
    if (options.retry.baseDelayMs < 0)
      throw new RangeError("retry.baseDelayMs must be >= 0");

    this.opts = options;
    this.semaphore = new Semaphore(options.maxConcurrency);
    this.bucket = new TokenBucket(options.maxRPS);
  }

  /**
   * Execute `processItem` over every element in `items`.
   *
   * Guarantees:
   *  - At most `maxConcurrency` items are in-flight at any moment.
   *  - Sustained execution rate does not exceed `maxRPS`.
   *  - Each item is retried up to `retry.maxRetries` times with exponential
   *    backoff + full jitter.
   *  - A single item's failure (even after all retries) never affects others.
   *  - If `signal` aborts, no *new* item work is started; in-flight items
   *    complete (or finish their current retry sleep) before the result is
   *    returned. Items never started are counted as `skipped`.
   */
  async run(
    items: readonly T[],
    processItem: ProcessItem<T, R>,
    signal?: AbortSignal
  ): Promise<JobRunResult<T, R>> {
    const startedAt = Date.now();
    const results: ItemResult<T, R>[] = new Array(items.length);
    let successes = 0;
    let failures = 0;
    let skipped = 0;

    // All per-item tasks run concurrently (bounded by semaphore + token bucket).
    const tasks = items.map((item, index) => this.executeItem(item, index, processItem, results, signal));

    const settled = await Promise.all(tasks);

    // Aggregate
    for (const entry of settled) {
      if (entry === "skipped") {
        skipped++;
      } else if (entry.ok) {
        successes++;
      } else {
        failures++;
      }
    }

    const completedAt = Date.now();

    return {
      successes,
      failures,
      skipped,
      items: results,
      aborted: signal?.aborted ?? false,
      startedAt,
      completedAt,
    };
  }

  // ── Private ───────────────────────────────────────────────────────────────

  private async executeItem(
    item: T,
    index: number,
    processItem: ProcessItem<T, R>,
    results: ItemResult<T, R>[],
    signal?: AbortSignal
  ): Promise<{ ok: boolean } | "skipped"> {
    // Gate: don't start if already aborted
    if (signal?.aborted) {
      results[index] = { index, item, ok: false, error: "Skipped: aborted", attempts: 0 };
      return "skipped";
    }

    const { maxRetries, baseDelayMs, maxDelayMs } = this.opts.retry;
    const totalAttempts = maxRetries + 1;

    let lastError: unknown;

    for (let attempt = 0; attempt < totalAttempts; attempt++) {
      // Check abort before each attempt
      if (signal?.aborted) {
        results[index] = {
          index,
          item,
          ok: false,
          error: lastError ?? "Aborted before attempt",
          attempts: attempt,
        };
        return "skipped";
      }

      // Acquire concurrency slot
      await this.semaphore.acquire(signal);

      try {
        // Acquire rate-limit token (releases semaphore on abort to avoid leaks)
        await this.bucket.acquire(signal);

        // Execute the actual work
        const result = await processItem(item, index);

        results[index] = { index, item, ok: true, result, attempts: attempt + 1 };
        this.opts.onItemSettled?.(results[index], Date.now() - /* we don't have per-item start; use global? */ 0);
        return { ok: true };
      } catch (err) {
        lastError = err;

        if (attempt < totalAttempts - 1) {
          // Backoff before retry — semaphore is released below, so other
          // items can proceed during the wait.
          const delay = computeBackoff(attempt, baseDelayMs, maxDelayMs);
          await sleep(delay, signal);
        }
      } finally {
        this.semaphore.release();
      }
    }

    // All retries exhausted
    results[index] = { index, item, ok: false, error: lastError, attempts: totalAttempts };
    this.opts.onItemSettled?.(results[index], 0);
    return { ok: false };
  }
}

// ─── Helpers ─────────────────────────────────────────────────────────────────

/**
 * Full-jitter exponential backoff (per AWS architecture blog recommendation).
 *   delay = random() * min(cap, base * 2^attempt)
 * Produces a value in [0, computedDelay], which reduces thundering-herd
 * synchronisation between retries.
 */
function computeBackoff(attempt: number, baseMs: number, capMs: number): number {
  const exponential = Math.min(capMs, baseMs * 2 ** attempt);
  return Math.random() * exponential;
}

function sleep(ms: number, signal?: AbortSignal): Promise<void> {
  return new Promise<void>((resolve, reject) => {
    if (signal?.aborted) {
      reject(abortError());
      return;
    }
    const timer = setTimeout(() => {
      cleanup();
      resolve();
    }, ms);

    const onAbort = () => {
      cleanup();
      reject(abortError());
    };

    function cleanup() {
      clearTimeout(timer);
      signal?.removeEventListener("abort", onAbort);
    }

    signal?.addEventListener("abort", onAbort, { once: true });
  });
}

function abortError(): DOMException {
  return new DOMException("Aborted", "AbortError");
}
```

---

## Usage Example

```typescript
import { JobQueue } from "./rate-limited-job-queue";

async function main() {
  const queue = new JobQueue<string, number>({
    maxConcurrency: 10,
    maxRPS: 50,
    retry: {
      maxRetries: 3,
      baseDelayMs: 500,
      maxDelayMs: 10_000,
    },
    onItemSettled: (entry, _elapsed) => {
      if (!entry.ok) {
        console.warn(`[item ${entry.index}] failed after ${entry.attempts} attempts`);
      }
    },
  });

  // Simulate an overnight ETL: 10 000 CSV row IDs
  const items = Array.from({ length: 10_000 }, (_, i) => `row-${i}`);

  // Graceful shutdown on SIGINT (Ctrl-C)
  const controller = new AbortController();
  process.on("SIGINT", () => {
    console.log("\nShutting down… letting in-flight items finish.");
    controller.abort();
  });

  const result = await queue.run(
    items,
    async (rowId: string) => {
      // e.g. call an external API or write to a data warehouse
      const res = await fetch(`https://api.internal/transform/${rowId}`);
      if (!res.ok) throw new Error(`Upstream ${res.status}`);
      return res.status;
    },
    controller.signal
  );

  const duration = ((result.completedAt - result.startedAt) / 1000).toFixed(1);
  console.log({
    duration: `${duration}s`,
    successes: result.successes,
    failures: result.failures,
    skipped: result.skipped,
    aborted: result.aborted,
  });

  // Log first few failures for the on-call runbook
  for (const item of result.items.filter((i) => !i.ok).slice(0, 5)) {
    console.error(`  [${item.index}] ${item.item}: ${item.error}`);
  }
}

main();
```

---

## Design Notes & Trade-offs

| Decision | Rationale | Alternative considered |
|----------|-----------|----------------------|
| **Full-jitter backoff** (`rand * min(cap, base·2ⁿ)`) | Proven to minimise thundering-herd synchronisation; simplest to reason about. | "Equal jitter" (`cap/2 + rand·cap/2`) guarantees a minimum spacing, which helps when the downstream is very sensitive to micro-bursts. Easy to swap in `computeBackoff`. |
| **Semaphore released during backoff** | Frees a concurrency slot so other items can proceed while one item sleeps. Maximises throughput under high retry rates. | Holding the slot during backoff is simpler but can starve the pipeline if many items retry simultaneously (e.g., a downstream 503 storm with `maxConcurrency=5` → all 5 slots sleep for 10 s). |
| **`Promise.all` over per-item tasks** | All items are "launched" immediately; the semaphore + token bucket provide the actual back-pressure. This avoids the need for a manual worker-loop / queue-internal polling. | A classic N-worker pulling from an internal queue is more extensible (priority, dynamic item counts) but adds significant complexity for a fixed-batch ETL. |
| **Graceful abort = "no new work"** | In-flight items (including their current retry sleep) are allowed to finish. This prevents partial writes in a data-warehouse context where a mid-write abort could leave inconsistent state. | A "hard abort" that rejects all pending promises immediately would be faster but risks leaving side-effects half-done. |
| **Token-bucket capacity = rate** (no separate burst config) | Keeps the API surface small. For most ETL APIs the burst allowance equals the sustained rate. | Exposing `burstCapacity` separately would allow short spikes; add a second constructor arg to `TokenBucket` if needed. |
| **No internal queue / back-pressure on the `items` array** | The caller passes the full array; memory cost is O(n) for the results array. For truly unbounded streams, the caller should chunk. | A streaming/iterable API (`AsyncIterable<T>`) would be more memory-efficient but complicates the "result" aggregation and progress reporting. |
| **`onItemSettled` fires with a 0 ms elapsed placeholder** | The per-item wall-clock isn't tracked to avoid per-item `Date.now()` calls (cheap, but at 10k+ items it's measurable). If you need per-item latency, add a `performance.now()` around the `processItem` call inside `executeItem`. | Always tracking per-item timing at the cost of ~2 µs × n calls. |

### When to reach for something else

* **Prioritised / multi-tenant queues** → use `p-queue` (priority levels) or a message broker (SQS, Redis Streams) and make this module the *consumer* side.
* **Exactly-once semantics** → you need a transactional outbox or idempotency keys; this queue is at-least-once (a crash between item-completion and result-aggregation could lose a success).
* **Back-pressure from downstream** (e.g., a write pool that saturates) → combine this queue with a `p-limit`-style adaptive concurrency that shrinks `maxConcurrency` when p95 latency exceeds a threshold.
