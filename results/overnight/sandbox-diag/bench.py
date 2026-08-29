#!/usr/bin/env python3
"""Time fib(35) and fib_memo(35)."""

import time

from fib_module import fib, fib_memo


def time_it(fn, n, runs=1000):
    best = float("inf")
    for _ in range(runs):
        fib_memo.cache_clear()  # measure real work, not cached lookups
        start = time.perf_counter()
        fn(n)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    t_fib = time_it(fib, 35)
    t_memo = time_it(fib_memo, 35)
    print(f"fib(35):    {t_fib * 1e6:.3f} µs")
    print(f"fib_memo(35): {t_memo * 1e6:.3f} µs")


if __name__ == "__main__":
    main()
