#!/usr/bin/env python3
"""CLI wrapper: print the nth Fibonacci number.

Usage: python3 fib.py 30
"""

import argparse

from fib_module import fib


def main() -> None:
    parser = argparse.ArgumentParser(description="Print the nth Fibonacci number")
    parser.add_argument("n", type=int, help="non-negative index")
    args = parser.parse_args()
    print(fib(args.n))


if __name__ == "__main__":
    main()
