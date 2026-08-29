import time

from fib_module import fib, fib_memo


def time_call(fn, n, runs=100):
    best = float("inf")
    for _ in range(runs):
        start = time.perf_counter()
        fn(n)
        best = min(best, time.perf_counter() - start)
    return best


if __name__ == "__main__":
    t_fib = time_call(fib, 35)
    t_memo = time_call(fib_memo, 35)
    print(f"fib(35):    {t_fib * 1000:.6f} ms")
    print(f"fib_memo(35): {t_memo * 1000:.6f} ms")
