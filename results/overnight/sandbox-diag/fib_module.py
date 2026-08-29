"""Fibonacci number implementations.

Exposes an iterative ``fib(n)`` and a memoized recursive ``fib_memo(n)``.
"""

from __future__ import annotations

from functools import lru_cache


def fib(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed) using an iterative loop."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@lru_cache(maxsize=None)
def fib_memo(n: int) -> int:
    """Return the nth Fibonacci number (0-indexed) using memoized recursion."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)
