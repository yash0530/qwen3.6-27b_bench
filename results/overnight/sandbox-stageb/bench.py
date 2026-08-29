"""Time fib(35) and fib_memo(35) and print both results."""

import time

from fib_module import fib, fib_memo


def time_call(func, n):
    start = time.perf_counter()
    result = func(n)
    return result, time.perf_counter() - start


if __name__ == "__main__":
    n = 35
    fib_result, fib_time = time_call(fib, n)
    memo_result, memo_time = time_call(fib_memo, n)
    print(f"fib({n}) = {fib_result} in {fib_time * 1e6:.1f} us")
    print(f"fib_memo({n}) = {memo_result} in {memo_time * 1e6:.1f} us")
