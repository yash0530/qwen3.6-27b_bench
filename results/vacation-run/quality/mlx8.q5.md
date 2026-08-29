# mlx8 — q5 `q5_debugging` (debugging)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=506.3s decode=9.7 tok/s prefill=345.3 tok/s gen_tokens=4908 prompt_tokens=216 finish=stop
- thinking chars: 13857

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

## Bug Analysis

### Bug 1 — No In-Flight Deduplication (Thundering Herd)

```ts
if (cache.has(key)) return cache.get(key)!;   // ← check
const data = await fetchData(key);            // ← YIELD: control returns to event loop
const p = Promise.resolve(data);
cache.set(key, p);                            // ← set (too late)
```

The `await` suspends the function and returns control to the event loop. Any other caller that invokes `load(sameKey)` during that suspension hits `cache.has(key)` → `false` (the entry hasn't been written yet) and fires a second `fetchData`. Under N concurrent callers you get N network calls instead of 1.

**Precise race:**

```
Timeline (single-threaded event loop):

tick 0  Caller A: cache.has("k") → false
tick 0  Caller A: fetchData("k") → returns Promise P_A
tick 0  Caller A: await → SUSPENDS (yields to loop)

tick 0  Caller B: cache.has("k") → false   ← P_A not yet in cache
tick 0  Caller B: fetchData("k") → returns Promise P_B  ← DUPLICATE CALL
tick 0  Caller B: await → SUSPENDS

... both P_A and P_B resolve, two network calls were made ...
```

### Bug 2 — Rejected Promises Cached Forever

This manifests the moment you naively fix Bug 1 by storing the promise *before* the `await`:

```ts
// Naive "fix" for Bug 1:
const promise = fetchData(key);
cache.set(key, promise);       // stored before we know the outcome
return promise;
```

If `fetchData` rejects, the rejected promise is permanently pinned in the `Map`. Every subsequent caller gets the same stale rejection. There is no retry path and no way to evict it.

### Bug 3 — Unnecessary `Promise.resolve` Wrapper

```ts
const data = await fetchData(key);  // `data` is a plain value, already unwrapped
const p = Promise.resolve(data);    // no-op: wraps a non-thenable in a resolved promise
```

This is not a correctness bug, but it signals a misunderstanding: the author stored the *resolved value* rather than the *in-flight promise*, which is the root cause of Bug 1.

### Bug 4 — `async` Wrapper Adds a Spurious Microtask

The function is `async` but never `await`s internally. The `return promise` inside an `async` function is equivalent to `return Promise.resolve(promise)`, introducing an extra microtask hop before the caller's `.then` fires. Negligible in most cases, but in a hot cache path it's avoidable latency.

---

## Corrected Implementation

```ts
const cache = new Map<string, Promise<Data>>();

/**
 * Deduplicates concurrent calls to fetchData for the same key.
 * - The in-flight promise is cached synchronously (before any await),
 *   so concurrent callers share one network request.
 * - On rejection the entry is evicted so the next caller can retry.
 * - No extra microtask is introduced (non-async function).
 */
function load(key: string): Promise<Data> {
  const existing = cache.get(key);
  if (existing !== undefined) return existing;

  // Attach rejection handler synchronously so:
  //   1. The source promise never goes unhandled.
  //   2. We evict the cache entry on failure (allowing retry).
  //   3. We re-throw so all callers still observe the rejection.
  const promise: Promise<Data> = fetchData(key).catch((err) => {
    cache.delete(key);
    throw err;
  });

  cache.set(key, promise);
  return promise;
}
```

### Why this is correct

| Concern | How it's handled |
|---|---|
| **Deduplication** | `cache.set(key, promise)` executes synchronously in the same microtask as `cache.get`. No `await` intervenes, so a concurrent caller in the same tick sees the entry. |
| **Rejection eviction** | The `.catch` handler deletes the key *before* re-throwing. The derived promise still rejects to all callers, but the cache is clean for the next attempt. |
| **Unhandled rejection safety** | `.catch` is attached to `fetchData(key)`'s return value *synchronously*, so the source promise always has a handler. The derived `promise` is returned to the caller who is contractually responsible for handling it. |
| **No extra microtask** | The function is not `async`; it returns the promise directly. Callers' `.then` callbacks fire one tick earlier than with an `async` wrapper. |
| **Synchronous throw in fetchData** | If `fetchData` throws synchronously, the `async`-less function would propagate it as a thrown error. Wrap in try/catch if that's a concern, or keep `async` for uniform rejection semantics. |

---

## Production Hardening (Optional Extensions)

```ts
interface CacheEntry {
  promise: Promise<Data>;
  createdAt: number;
}

const cache = new Map<string, CacheEntry>();
const TTL = 60_000; // 60 s

function load(key: string): Promise<Data> {
  const existing = cache.get(key);
  if (existing !== undefined) {
    // Lazy TTL check (avoids timers, works with frozen clocks)
    if (Date.now() - existing.createdAt < TTL) {
      return existing.promise;
    }
    cache.delete(key); // expired → fall through to reload
  }

  const promise: Promise<Data> = fetchData(key).catch((err) => {
    // Evict only if this is still the current entry (avoid deleting a
    // newer entry that was set after a retry)
    const entry = cache.get(key);
    if (entry?.promise === promise) {
      cache.delete(key);
    }
    throw err;
  });

  cache.set(key, { promise, createdAt: Date.now() });
  return promise;
}
```

The guard `entry?.promise === promise` in the eviction path handles the (rare) race where a new `load(key)` has already overwritten the cache entry by the time the old promise rejects—preventing the new entry from being incorrectly evicted.

---

## Summary of Root Causes

1. **Temporal gap between check and set** straddles an `await`, breaking the check-then-act invariant.
2. **Storing the promise before knowing its outcome** without a rejection-eviction handler creates a poisoned-cache scenario.
3. The fix is to **close the temporal gap** (no `await` between check and set) **and** **attach a self-cleaning rejection handler** to the cached promise.
