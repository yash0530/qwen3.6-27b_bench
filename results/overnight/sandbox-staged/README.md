# Fibonacci

A tiny Fibonacci library: an iterative implementation plus a memoized recursive
variant, with a command-line wrapper.

## Files

- `fib_module.py` — the library. Exposes `fib(n)` and `fib_memo(n)`.
- `fib.py` — CLI wrapper around the module.
- `test_fib.py` — pytest test suite.
- `bench.py` — benchmarks `fib(35)` vs `fib_memo(35)`.

## CLI

`fib.py` takes a single positional argument `n` and prints `fib(n)`:

```console
$ python3 fib.py 30
832040

$ python3 fib.py -h
usage: fib.py [-h] n

Compute a Fibonacci number

positional arguments:
  n         index of the Fibonacci number

options:
  -h, --help  show this help message and exit
```

## Module

Both functions return the n-th Fibonacci number, with `fib(0) == 0` and
`fib(1) == 1`. They agree for all `n >= 0`; they differ only in how they get
there.

```python
from fib_module import fib, fib_memo

fib(10)        # 55 — iterative, O(n) time and space
fib_memo(10)   # 55 — recursive with memoization, O(n) time (space per process)
```

- `fib(n)` is a plain loop; no allocation beyond two running values, and no
  recursion-limit concerns for large `n`.
- `fib_memo(n)` is the classic recursive definition with a memo cache that is
  created per top-level call (reuse one across calls by passing
  `fib_memo(n, cache=shared)`). Recursion depth is bounded by Python's
  recursion limit, so very large `n` can hit it.

Both accept a non-negative integer `n`; a negative `n` raises `ValueError`.

## Tests

```console
$ python3 -m pytest
```

## Benchmarks

`bench.py` times one call each of `fib(35)` and `fib_memo(35)`:

```console
$ python3 bench.py
fib(35):    1.458 us
fib_memo(35): 15.208 us
```

(The `fib_memo` call includes building a fresh memo cache, so expect it to be
several times slower on a cold call; numbers vary by machine.)
