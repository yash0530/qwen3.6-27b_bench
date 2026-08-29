"""Time fib(35) and fib_memo(35).

Usage: python3 bench.py
"""
import time

from fib_module import fib, fib_memo


def time_it(func, n: int, repeats: int = 5) -> float:
    """Return the best (seconds) over a few timed calls."""
    best = float("inf")
    for _ in range(repeats):
        start = time.perf_counter()
        func(n)
        best = min(best, time.perf_counter() - start)
    return best


def main() -> None:
    t_iter = time_it(fib, 35)
    t_memo = time_it(fib_memo, 35)
    print(f"fib(35)      (iterative):       {t_iter:.6f}s")
    print(f"fib_memo(35) (memoized rec.):  {t_memo:.6f}s")


if __name__ == "__main__":
    main()
