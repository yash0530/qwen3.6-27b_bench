# fib

A tiny Fibonacci number toolkit: a command-line interface plus a reusable module.

## CLI

`fib.py` prints the n-th Fibonacci number (0-indexed), where `n` is a required positional argument.

```bash
python3 fib.py 30
# 832040
```

## Module

`fib_module.py` exposes two functions:

- `fib(n)` — iterative O(n) computation.
- `fib_memo(n)` — memoized recursive computation (same result, different strategy).

```python
from fib_module import fib, fib_memo

fib(10)      # 55
fib_memo(10) # 55
```

Both functions raise `ValueError` for negative input.

## Tests

Run the test suite with pytest:

```bash
python3 -m pytest test_fib.py -q
```

## Benchmark

`bench.py` times `fib(35)` and `fib_memo(35)` and prints both timings:

```bash
python3 bench.py
```

Example output (single run):

```text
fib(35):     1.583 us
fib_memo(35): 8.292 us
```

The iterative `fib` is faster here: it's a tight loop with no call overhead, while `fib_memo` pays recursion cost on a fresh (per-call) memo. Timings vary per run.
