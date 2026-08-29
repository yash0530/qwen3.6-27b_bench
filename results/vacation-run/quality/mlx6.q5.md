# mlx6 — q5 `q5_debugging` (debugging)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=482.1s decode=12.4 tok/s prefill=318.9 tok/s gen_tokens=5985 prompt_tokens=216 finish=stop
- thinking chars: 12243

## Prompt

The following TypeScript is meant to cache the result of an async loader so concurrent callers share one in-flight request, but under load it sometimes makes duplicate network calls and occasionally caches a rejected promise forever. Find every bug, explain the race conditions precisely, and provide a corrected, production-quality version.

```ts
const cache = new Map<string, Promise<Data>>();
async function load(key: string): Promise<Data> {
  if (cache.has(key)) return cache.get(key)!;
  const data = await fetchData(key);
  const p = Promise.resolve(data);
  cache.set(key, p);
  return data;
}
```

## Answer

# Bug Analysis: Async Cache with In-Flight Deduplication

## The Code Under Scrutiny

```ts
const cache = new Map<string, Promise<Data>>();
async function load(key: string): Promise<Data> {
  if (cache.has(key)) return cache.get(key)!;
  const data = await fetchData(key);
  const p = Promise.resolve(data);
  cache.set(key, p);
  return data;
}
```

---

## Bug 1 — No In-Flight Deduplication (Duplicate Network Calls)

**Root cause:** `cache.set(key, p)` executes *after* the `await`, so the in-flight request is invisible to concurrent callers.

**Precise race timeline** (single-threaded event loop, two concurrent callers):

```
─────────────────────────────────────────────────────────────────────
  Caller A                              Caller B
─────────────────────────────────────────────────────────────────────
load("x") called
  cache.has("x") → false
  fetchData("x") → returns P₁
  await P₁ ──────────── suspends ──┐
                                    │  P₁ in flight (network)
                                   │
                    load("x") called
                      cache.has("x") → false  ← P₁ not stored yet
                      fetchData("x") → returns P₂   ← DUPLICATE CALL
                      await P₂ ────── suspends ──┐
                                                  │  P₁ and P₂ both in flight
─────────────────────────────────────────────────────────────────────
P₁ resolves with data₁
  resume Caller A
  cache.set("x", resolve(data₁))
  return data₁
─────────────────────────────────────────────────────────────────────
P₂ resolves with data₂
  resume Caller B
  cache.set("x", resolve(data₂))   ← overwrites entry
  return data₂
─────────────────────────────────────────────────────────────────────
```

**Consequences:**
- N concurrent callers → N network calls (thundering herd).
- Under bursty load this can cascade: the N responses arrive, N−1 are discarded, but the N requests already consumed bandwidth, hit rate limits, or mutated server state.

---

## Bug 2 — The "Obvious Fix" Caches Rejections Forever

The natural fix for Bug 1 is to store the promise *before* awaiting:

```ts
// Naive "fix" — introduces Bug 2
const p = fetchData(key);
cache.set(key, p);   // in-flight promise is now visible ✓
return p;             // but if p rejects, it's cached as a tombstone ✗
```

Once `fetchData(key)` rejects, the rejected promise sits in the `Map` permanently. Every subsequent `load(key)` call returns the same rejected promise, making the failure **perpetual** until the process is restarted.

**Why this is severe:**
- Transient failures (503, timeout, DNS blip) become permanent.
- No retry path exists; the cache is now a poison pill.
- If the rejection is an unhandled-rejection in some code path, it can crash the Node.js process.

---

## Bug 3 — Unbounded Memory Growth

The `Map` has no eviction policy. If `key` is drawn from an unbounded domain (URLs, user-supplied identifiers, composite query strings), the cache grows without limit → OOM.

---

## Bug 4 — No TTL / Staleness Control

Even for successful results, the cached value is immortal. A backend data change is invisible to all callers until process restart.

---

## Bug 5 — No Negative-Cache / Circuit-Breaker Strategy

There is no way to express "remember this failure for 5 seconds to prevent hammering a dead backend." The binary choice is either *never* cache failures (retry storms) or *forever* cache them (Bug 2).

---

## Corrected, Production-Quality Implementation

```ts
// ─── Types ───────────────────────────────────────────────────────────────────

interface CacheEntry<T> {
  promise: Promise<T>;
  /** Epoch ms when the entry was created (for TTL + LRU eviction). */
  createdAt: number;
}

interface DataLoaderOptions {
  /** Max age of a successful entry before it's considered stale. Default: 60 s. */
  ttlMs?: number;
  /** Max number of entries before LRU eviction kicks in. Default: 10 000. */
  maxEntries?: number;
  /**
   * After a rejection, how long to suppress retries (ms).
   * 0 = retry immediately on next call. Default: 5 000.
   */
  negativeCacheMs?: number;
}

// ─── Implementation ──────────────────────────────────────────────────────────

class DataLoader<T> {
  private readonly cache = new Map<string, CacheEntry<T>>();
  /** Keys currently in a negative (post-failure) cooldown. */
  private readonly negativeCache = new Map<string, number>(); // key → expiryMs

  private readonly ttlMs: number;
  private readonly maxEntries: number;
  private readonly negativeCacheMs: number;

  constructor(
    private readonly fetcher: (key: string) => Promise<T>,
    opts: DataLoaderOptions = {},
  ) {
    this.ttlMs = opts.ttlMs ?? 60_000;
    this.maxEntries = opts.maxEntries ?? 10_000;
    this.negativeCacheMs = opts.negativeCacheMs ?? 5_000;
  }

  /**
   * Load data for `key`, sharing a single in-flight request among
   * all concurrent callers.
   */
  load(key: string): Promise<T> {
    // 1. Fast path: hit a live, non-expired positive cache entry.
    const entry = this.cache.get(key);
    if (entry && !this.isExpired(entry)) {
      return entry.promise;
    }

    // 2. If entry exists but is expired, remove it.
    if (entry) {
      this.cache.delete(key);
    }

    // 3. Negative cache: suppress retries during cooldown after a failure.
    const negExpiry = this.negativeCache.get(key);
    if (negExpiry && Date.now() < negExpiry) {
      // Return the stored rejection (or a fresh one — design choice).
      // Here we throw immediately to signal "still cooling down."
      return Promise.reject(new DataLoaderError(
        `Key "${key}" is in negative-cache cooldown until ${new Date(negExpiry).toISOString()}`,
      ));
    }
    // Clean up stale negative entries.
    if (negExpiry) this.negativeCache.delete(key);

    // 4. Evict to make room if at capacity.
    if (this.cache.size >= this.maxEntries) {
      this.evictOldest();
    }

    // 5. ★ Single-flight: store the promise BEFORE any await.
    //    All concurrent callers will see this same in-flight promise.
    const promise = this.fetchWithCleanup(key);
    const newEntry: CacheEntry<T> = { promise, createdAt: Date.now() };
    this.cache.set(key, newEntry);

    return promise;
  }

  // ─── Private helpers ──────────────────────────────────────────────────────

  /**
   * Wraps the raw fetcher so that on rejection we:
     a) evict the entry from the positive cache (allowing future retry),
     b) record a negative-cache cooldown.
   *
   * The returned promise is what gets stored in the cache and returned
   * to all concurrent callers.
   */
  private fetchWithCleanup(key: string): Promise<T> {
    return this.fetcher(key).then(
      (data) => {
        // Success: the entry stays in the cache with its TTL.
        // (Nothing to do — entry was already stored by the caller.)
        return data;
      },
      (err) => {
        // Failure: evict + set negative cooldown.
        this.cache.delete(key);
        this.negativeCache.set(key, Date.now() + this.negativeCacheMs);
        throw err; // re-throw so every concurrent caller sees the rejection
      },
    );
  }

  private isExpired(entry: CacheEntry<T>): boolean {
    return Date.now() - entry.createdAt >= this.ttlMs;
  }

  /**
   * Remove the oldest entry (by `createdAt`).
   * O(n) scan — acceptable for maxEntries ≈ 10k.
   * For larger caches, use a doubly-linked list or a secondary LRU structure.
   */
  private evictOldest(): void {
    let oldestKey: string | null = null;
    let oldestTime = Infinity;
    for (const [k, v] of this.cache) {
      if (v.createdAt < oldestTime) {
        oldestTime = v.createdAt;
        oldestKey = k;
      }
    }
    if (oldestKey !== null) {
      this.cache.delete(oldestKey);
    }
  }

  /** Invalidate a single key (e.g., on a known mutation). */
  invalidate(key: string): void {
    this.cache.delete(key);
    this.negativeCache.delete(key);
  }

  /** Invalidate all keys. */
  clear(): void {
    this.cache.clear();
    this.negativeCache.clear();
  }
}

// ─── Error type ──────────────────────────────────────────────────────────────

class DataLoaderError extends Error {
  constructor(message: string) {
    super(message);
    this.name = "DataLoaderError";
  }
}

// ─── Usage ───────────────────────────────────────────────────────────────────

const loader = new DataLoader<string>(
  (key) => fetch(`/api/${key}`).then((r) => {
    if (!r.ok) throw new Error(`HTTP ${r.status}`);
    return r.text();
  }),
  { ttlMs: 30_000, maxEntries: 5_000, negativeCacheMs: 10_000 },
);

// All three concurrent calls share ONE network request:
const [a, b, c] = await Promise.all([
  loader.load("users/42"),
  loader.load("users/42"),
  loader.load("users/42"),
]);
```

---

## Why Each Design Decision Matters

| Decision | Problem it solves |
|---|---|
| **Store promise before `await`** (step 5) | Single-flight dedup — all concurrent callers attach to the same in-flight promise. |
| **Evict on rejection + negative cooldown** | Prevents both the "poison pill" (Bug 2) *and* the "retry storm" on a dead backend. |
| **TTL on positive entries** | Bounds staleness (Bug 4). |
| **Max-entries LRU eviction** | Bounds memory (Bug 3). |
| **`then(onFulfilled, onRejected)` instead of `try/catch` around `await`** | The stored promise in the cache *is* the one with the cleanup handler attached. No intermediate `await` means the cleanup is guaranteed to run exactly once, even if zero external callers ever `await` the promise (prevents unhandled-rejection edge cases). |
| **`invalidate()` / `clear()`** | Lets the application layer react to known mutations without waiting for TTL expiry. |

---

## Edge-Case Notes

1. **Unhandled rejections:** Because the stored promise has a `.then(_, onRejected)` handler attached, Node's unhandled-rejection detection is satisfied even if no external caller ever awaits the promise. The `throw err` inside `onRejected` creates a *new* rejected promise that is the one callers receive—each caller must handle it, which is correct.

2. **Idempotency assumption:** This pattern is safe only when `fetcher(key)` is idempotent (GET-like). For mutating operations, you would want a different concurrency model (mutex, queue).

3. **Scalability beyond single process:** The `Map` is per-process. For a multi-instance deployment, replace with a shared store (Redis, in-memory pub/sub) or accept that each process makes its own in-flight request (usually fine because the N+1 problem is per-process).

4. **`maxEntries` eviction is O(n):** For caches > ~100k entries, swap the `Map` for a proper LRU (doubly-linked list + Map) or use a library like `lru-cache` with a custom `fetchMethod`.
