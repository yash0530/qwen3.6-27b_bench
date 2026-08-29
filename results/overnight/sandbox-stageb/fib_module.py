"""Fibonacci number functions."""

from __future__ import annotations

from typing import Dict


def fib(n: int) -> int:
    """Return the nth Fibonacci number (fib(0) = 0, fib(1) = 1), computed iteratively."""
    a: int = 0
    b: int = 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_memo(n: int) -> int:
    """Return the nth Fibonacci number, computed recursively with memoization."""
    if n < 0:
        raise ValueError("n must be non-negative")
    cache: Dict[int, int] = {}

    def _fib(k: int) -> int:
        if k < 2:
            return k
        if k not in cache:
            cache[k] = _fib(k - 1) + _fib(k - 2)
        return cache[k]

    return _fib(n)
