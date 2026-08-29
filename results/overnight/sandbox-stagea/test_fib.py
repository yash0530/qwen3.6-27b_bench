import pytest

from fib_module import fib, fib_memo


def test_fib_0():
    assert fib(0) == 0


def test_fib_1():
    assert fib(1) == 1


@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_matches_memo(n):
    assert fib(n) == fib_memo(n)
