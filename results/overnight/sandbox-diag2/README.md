# Fibonacci Tools

A small project providing two Fibonacci implementations, a CLI, tests, and a benchmark.

## Layout

- `fib_module.py` — the core logic: `fib(n)` (iterative) and `fib_memo(n)` (memoized recursive)
- `fib.py` — CLI wrapper: prints the nth Fibonacci number
- `test_fib.py` — pytest suite
- `bench.py` — times `fib(35)` vs `fib_memo(35)`

## CLI usage

```
$ python3 fib.py 30
832040

$ python3 fib.py 10
55
```

## Module usage

```python
from fib_module import fib, fib_memo

fib(10)      # 55  — iterative, O(n) time / O(1) space
fib_memo(10) # 55  — memoized recursive, O(n) time / O(n) space
```

Both require `n >= 0` and raise `ValueError` for negative input.

## Tests

```
$ python3 -m pytest test_fib.py
9 passed
```

## Benchmark

```
$ python3 bench.py
fib(35)      (iterative):       0.000001s
fib_memo(35) (memoized rec.):  0.000000s
```

(Each figure is the best of 5 timed calls; at n=35 both are so fast they hover at
the timer resolution, with the memoized variant marginally faster.)
