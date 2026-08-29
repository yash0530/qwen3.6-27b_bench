import pytest

from fib_module import fib, fib_memo

EXPECTED = {0: 0, 1: 1, 10: 55, 20: 6765, 30: 832040}


def test_fib_0():
    assert fib(0) == 0


def test_fib_1():
    assert fib(1) == 1


def test_fib_10():
    assert fib(10) == 55


@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_and_fib_memo_agree(n):
    assert fib(n) == EXPECTED[n]
    assert fib_memo(n) == EXPECTED[n]
    assert fib(n) == fib_memo(n)
