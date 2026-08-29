"""Time one call each of fib(35) and fib_memo(35)."""

import time

from fib_module import fib, fib_memo


def main():
    start = time.perf_counter()
    fib(35)
    fib_time = time.perf_counter() - start

    start = time.perf_counter()
    fib_memo(35)
    memo_time = time.perf_counter() - start

    print(f"fib(35):    {fib_time * 1e6:.3f} us")
    print(f"fib_memo(35): {memo_time * 1e6:.3f} us")


if __name__ == "__main__":
    main()
