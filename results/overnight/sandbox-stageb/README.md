# fib

A small Fibonacci toolkit: a command-line interface and a Python module.

## CLI

`fib.py` prints the nth Fibonacci number, given as a positional argument:

```
$ python3 fib.py 30
832040
```

## Module

`fib_module.py` exposes two functions:

```python
from fib_module import fib, fib_memo

fib(30)       # 832040 — iterative, O(n) time, O(1) space
fib_memo(30)  # 832040 — recursive with memoization, O(n) time, O(n) space
```

- `fib(n)` — computes the nth Fibonacci number iteratively (`fib(0) = 0`,
  `fib(1) = 1`).
- `fib_memo(n)` — same values, computed via recursion with a memoization
  cache; raises `ValueError` for negative `n`.

## Tests

```
$ python3 -m pytest test_fib.py
```

## Example output

`bench.py` (run on an M-series Mac, single run):

```
$ python3 bench.py
fib(35) = 9227465 in 1.4 us
fib_memo(35) = 9227465 in 7.7 us
```

The iterative version is faster for a single call because it avoids recursion and
dict-lookup overhead; `fib_memo` pays for the cache on the way up.
