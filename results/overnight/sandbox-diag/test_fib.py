import pytest

from fib_module import fib, fib_memo


@pytest.mark.parametrize("n, expected", [
    (0, 0),
    (1, 1),
    (10, 55),
])
def test_fib(n, expected):
    assert fib(n) == expected


@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_matches_memo(n):
    assert fib(n) == fib_memo(n)


def test_memo_known_values():
    assert fib_memo(0) == 0
    assert fib_memo(1) == 1
