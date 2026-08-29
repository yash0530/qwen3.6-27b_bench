# mlx4 — q5 `q5_debugging` (debugging)

- temp=1.0 top_p=0.95 max_tokens=8192 seed=1234 thinking=on effort=low APC=0 SPEC=0
- wall=291.3s decode=17.6 tok/s prefill=384.5 tok/s gen_tokens=5120 prompt_tokens=216 finish=stop
- thinking chars: 13272

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

# Bug Analysis

## Bug 1 — The Check-Then-Act Race (Duplicate Network Calls)

```ts
if (cache.has(key)) return cache.get(key)!;   // ① check
const data = await fetchData(key);            // ② SUSPEND
const p = Promise.resolve(data);
cache.set(key, p);                           // ③ populate
```

The `await` at ② is a **suspension point**. Control returns to the event loop, and any other microtask or macrotask that calls `load(key)` *before* ② resolves will also pass the `cache.has(key)` check at ① (the cache was never populated because ③ hasn't executed yet). Both callers then each invoke `fetchData(key)`.

**Concrete interleaving:**

```
Time ──────────────────────────────────────────────────────►

Caller A:  has(key)? → false ──► fetchData(key) ──[suspends]──► set(key, pA)
Caller B:              has(key)? → false ──► fetchData(key) ──[suspends]──► set(key, pB)
                                                   │
                                          second network call fires
```

Both A and B fire independent network requests. The second `cache.set` overwrites the first, so one in-flight promise is orphaned (its result is lost, but the HTTP request still completed).

---

## Bug 2 — Rejected Promises Never Cached *or* the Naive Fix Caches Them Forever

In the code as written, if `fetchData(key)` rejects, the `await` throws and execution **never reaches** `cache.set`. So the rejection is *not* cached — the next caller retries. That's arguably correct, but it means you get no single-flight deduplication on retries after a failure (every failed call fans out).

The **naive fix** for Bug 1 — moving `cache.set` before the `await`:

```ts
const p = fetchData(key);
cache.set(key, p);       // now set before suspension ✓
return p;                // but if p rejects, it stays in the Map forever ✗
```

— introduces the second symptom you described: the **rejected promise is permanently cached**. Every subsequent caller receives the same stale rejection and the key is never re-fetched. In a long-lived process this is a silent data-staleness / permanent-failure bug.

---

## Bug 3 — Redundant / Unsafe Double Lookup

```ts
if (cache.has(key)) return cache.get(key)!;
```

Two map lookups where one suffices, plus a non-null assertion that papers over the type system. In a single-threaded runtime there's no *race* between `has` and `get`, but the pattern is fragile (e.g., if someone later inserts a `setTimeout` or `await` between them for logging) and the `!` suppresses the compiler's ability to catch a future `undefined` value.

---

## Bug 4 — Unbounded Cache Growth (Design Flaw)

The `Map` is never pruned. Every distinct key that ever loads accumulates a promise (and therefore its resolved `Data`) forever. In a long-running service with high-cardinality keys this is a **slow memory leak**.

---

## Bug 5 — `async` Wrapper Adds an Unnecessary Microtask

Declaring `load` as `async` means `return p` (where `p` is already a `Promise`) causes the engine to internally `await p`, inserting an extra microtask hop before the caller's `.then` fires. For a hot path this is measurable latency for zero benefit.

---

# Corrected Production-Quality Version

```ts
type Data = { /* your shape */ };

const inflight = new Map<string, Promise<Data>>();

/**
 * Single-flight loader with rejection-aware caching.
 *
 * Guarantees:
 *  - At most ONE in-flight `fetchData(key)` per key at any time.
 *  - Rejected results are NOT cached; the next call retries.
 *  - Resolved results remain cached (add TTL below if needed).
 *  - No `async` wrapper → no extra microtask hop.
 */
export function load(key: string): Promise<Data> {
  // Single lookup; a Promise is always an object, so `!== undefined`
  // is the precise "not present" test (no false negatives if Data is nullish).
  const existing = inflight.get(key);
  if (existing !== undefined) return existing;

  // Kick off the request synchronously; we hold the promise handle
  // BEFORE any suspension point, making check→set atomic in the
  // single-threaded event loop.
  let p: Promise<Data>;
  try {
    p = fetchData(key);
  } catch (err) {
    // fetchData threw synchronously (only possible if it's not `async`).
    // Nothing was cached, so just propagate.
    return Promise.reject(err);
  }

  // Attach a side-effect handler for cleanup on rejection.
  // `void` signals intentional discard of the derived promise,
  // suppressing "floating promise" lints and unhandled-rejection warnings.
  void p.catch(() => {
    // Only delete if this is still the current entry (a newer call
    // for the same key may have already replaced it).
    if (inflight.get(key) === p) {
      inflight.delete(key);
    }
  });

  inflight.set(key, p);
  return p;
}
```

### Why this is race-free

The critical invariant: **between `inflight.get(key)` and `inflight.set(key, p)` there is no `await`, no `.then`, and no other suspension point.** JavaScript's event loop guarantees that no other task (microtask or macrotask) can execute in that interval. Therefore:

| Caller | Sees `get(key)` | Sees `set(key, p)` |
|--------|-----------------|---------------------|
| A (first) | `undefined` → proceeds | populates Map |
| B (concurrent, arrives after A's `set`) | `p` → returns it | — |
| C (concurrent, arrives *during* A's `fetchData` pending) | `p` (already set synchronously) → returns it | — |

C is the case that the original code got wrong: C arrives after A's `await` suspends but before it resolves. In the corrected code, A has *already* called `inflight.set` before suspending, so C sees the in-flight promise and shares it.

### Rejection handling

The `.catch` handler deletes the entry **only if the stored promise is still `p`** (identity check). This handles the edge case where a rapid sequence of failures and retries causes a newer promise to have already overwritten the slot — we don't want to delete the *newer* entry.

---

## Optional Production Hardening

```ts
import { LRUCache } from "lru-cache";

// TTL + max-size eviction for resolved entries
const cache = new LRUCache<string, Promise<Data>>({
  max: 10_000,
  ttl: 5 * 60 * 1000, // 5 minutes
});

function load(key: string): Promise<Data> {
  const existing = cache.get(key);
  if (existing !== undefined) return existing;

  let p: Promise<Data>;
  try {
    p = fetchData(key);
  } catch (err) {
    return Promise.reject(err);
  }

  void p.catch(() => cache.delete(key)); // reject → evict immediately

  cache.set(key, p);
  return p;
}
```

The LRU cache handles:
- **Unbounded growth** (`max` evicts least-recently-used entries).
- **Stale data** (`ttl` expires resolved promises; on access after TTL the entry is treated as absent, triggering a fresh fetch).
- **Rejection cleanup** still works identically.

> **Note on TTL + single-flight interaction:** Once a TTL'd entry expires, `cache.get` returns `undefined`, the next caller creates a *new* promise and `cache.set` replaces the old one. Because `set` is synchronous, the single-flight guarantee still holds for that new in-flight request.
