import pytest

from fib_module import fib, fib_memo


def test_fib_0():
    assert fib(0) == 0


def test_fib_1():
    assert fib(1) == 1


@pytest.mark.parametrize("n,expected", [(0, 0), (1, 1), (10, 55)])
def test_fib_values(n: int, expected: int) -> None:
    assert fib(n) == expected


@pytest.mark.parametrize("n", [10, 20, 30])
def test_fib_matches_memo(n: int) -> None:
    assert fib(n) == fib_memo(n)


def test_negative_raises() -> None:
    with pytest.raises(ValueError):
        fib(-1)
    with pytest.raises(ValueError):
        fib_memo(-1)
