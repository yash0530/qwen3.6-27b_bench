from typing import Dict, Optional


def fib(n: int) -> int:
    """Return the n-th Fibonacci number iteratively (0-indexed)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    a, b = 0, 1
    for _ in range(n):
        a, b = b, a + b
    return a


def fib_memo(n: int, memo: Optional[Dict[int, int]] = None) -> int:
    """Return the n-th Fibonacci number via memoized recursion (0-indexed)."""
    if n < 0:
        raise ValueError("n must be non-negative")
    if memo is None:
        memo = {}
    if n not in memo:
        memo[n] = fib_memo(n - 1, memo) + fib_memo(n - 2, memo) if n >= 2 else n
    return memo[n]
