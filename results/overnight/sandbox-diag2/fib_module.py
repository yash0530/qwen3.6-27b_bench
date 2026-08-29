"""Fibonacci implementations.

Exposes:
    fib(n)      -- iterative
    fib_memo(n) -- memoized recursive
"""
from __future__ import annotations

from functools import cache


def fib(n: int) -> int:
    """Iterative Fibonacci: fib(0)=0, fib(1)=1.

    Runs in O(n) time and O(1) space.
    """
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


@cache
def fib_memo(n: int) -> int:
    """Memoized recursive Fibonacci, O(n) time / O(n) space via the cache."""
    if n < 0:
        raise ValueError("n must be a non-negative integer")
    if n < 2:
        return n
    return fib_memo(n - 1) + fib_memo(n - 2)
