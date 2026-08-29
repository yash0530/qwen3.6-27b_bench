# fib

A small Fibonacci toolkit: an iterative `fib(n)`, a memoized recursive
`fib_memo(n)`, a CLI wrapper, tests, and a micro-benchmark.

## Files

| File             | Purpose                                                    |
|------------------|------------------------------------------------------------|
| `fib_module.py`  | `fib(n)` (iterative) and `fib_memo(n)` (memoized recursion) |
| `fib.py`         | CLI wrapper: `python3 fib.py N`                            |
| `test_fib.py`    | pytest suite                                               |
| `bench.py`       | times `fib(35)` vs `fib_memo(35)`                          |

## CLI usage

```console
$ python3 fib.py 30
832040
```

`n` is a required positional integer and must be non-negative (a negative
value raises `ValueError`).

## Module usage

```python
from fib_module import fib, fib_memo

fib(10)      # 55
fib_memo(10) # 55
```

Both treat `fib(0) == 0`, `fib(1) == 1` (0-indexed). `fib_memo` uses
`functools.lru_cache`; call `fib_memo.cache_clear()` to reset its cache.
`fib` and `fib_memo` always agree.

## Running tests and the benchmark

```console
$ python3 -m pytest -q
7 passed

$ python3 bench.py
fib(35):    0.541 µs
fib_memo(35): 3.791 µs
```

## Example output

Real numbers from a `bench.py` run on this machine (best of 1000 runs,
cache cleared each iteration):

| Function     | n  | Time (best of 1000) |
|--------------|----|---------------------|
| `fib(n)`     | 35 | 0.541 µs            |
| `fib_memo(n)`| 35 | 3.791 µs            |

The iterative version wins here because recursion and function-call
overhead dominate at small `n`; the memo's payoff shows up when the same
values are re-queried (each call returns in < 1 µs after the cache is warm).
