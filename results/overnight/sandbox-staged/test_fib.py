import pytest

from fib_module import fib, fib_memo


def test_fib_0():
    assert fib(0) == 0


def test_fib_1():
    assert fib(1) == 1


def test_fib_10():
    assert fib(10) == 55


@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_memo_agrees_with_fib(n):
    assert fib_memo(n) == fib(n)
