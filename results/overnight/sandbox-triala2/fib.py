import argparse

from fib_module import fib


def main():
    parser = argparse.ArgumentParser(description="Print the n-th Fibonacci number.")
    parser.add_argument("n", type=int, help="the index of the Fibonacci number to compute")
    args = parser.parse_args()
    print(fib(args.n))


if __name__ == "__main__":
    main()
