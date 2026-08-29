# Fibonacci

A small Fibonacci number project with a CLI, a library module, and tests.

## Files

- `fib_module.py` — the core logic:
  - `fib(n)` — iterative Fibonacci, O(n) time, O(1) memory.
  - `fib_memo(n)` — memoized recursive variant (via `functools.lru_cache`).
- `fib.py` — command-line wrapper that imports from `fib_module`.
- `test_fib.py` — pytest suite.
- `bench.py` — times `fib(35)` vs `fib_memo(35)`.

## CLI usage

```bash
python3 fib.py 30
# 832040
```

The index `n` is a single positional argument.

## Module usage

```python
from fib_module import fib, fib_memo

fib(10)      # 55
fib_memo(10) # 55 (same result; values are memoized across calls)
```

## Running the tests

```bash
python3 -m pytest test_fib.py
```

## Benchmark

```bash
python3 bench.py
```

Prints the best-of-100 wall-clock timing of `fib(35)` and `fib_memo(35)` in milliseconds.

## Example output

```text
fib(35):    0.000666 ms
fib_memo(35): 0.000041 ms
```

(From an actual run; the memoized variant wins at this size because its cache is warm from a prior call, while `fib` always recomputes its 35-step loop.)
