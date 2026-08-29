"""Tests for the Fibonacci implementation."""

import pytest

from fib_module import fib, fib_memo


@pytest.mark.parametrize(
    "n, expected",
    [
        (0, 0),
        (1, 1),
        (10, 55),
    ],
)
def test_fib(n, expected):
    assert fib(n) == expected


@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_and_fib_memo_agree(n):
    assert fib(n) == fib_memo(n)
