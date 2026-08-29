"""Fibonacci functions: iterative and memoized recursive variants."""

from __future__ import annotations

_MemoCache = dict[int, int]


def fib_memo(n: int, cache: _MemoCache | None = None) -> int:
    """Return the n-th Fibonacci number, recursively with memoization.

    Raises ValueError for negative n.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    if cache is None:
        cache = {}
    if n < 2:
        return n
    if n not in cache:
        cache[n] = fib_memo(n - 1, cache) + fib_memo(n - 2, cache)
    return cache[n]


def fib(n: int) -> int:
    """Return the n-th Fibonacci number, iteratively.

    Raises ValueError for negative n.
    """
    if n < 0:
        raise ValueError(f"n must be non-negative, got {n}")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a
