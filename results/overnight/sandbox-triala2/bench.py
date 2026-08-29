import time

from fib_module import fib, fib_memo


def time_call(func, n):
    start = time.perf_counter()
    func(n)
    return time.perf_counter() - start


def main():
    fib_time = time_call(fib, 35)
    memo_time = time_call(fib_memo, 35)
    print(f"fib(35):    {fib_time * 1e6:.3f} us")
    print(f"fib_memo(35): {memo_time * 1e6:.3f} us")


if __name__ == "__main__":
    main()
