import argparse

from fib_module import fib


def main():
    parser = argparse.ArgumentParser(description="Print the nth Fibonacci number.")
    parser.add_argument("n", type=int, help="index of the Fibonacci number to compute")
    args = parser.parse_args()
    print(fib(args.n))


if __name__ == "__main__":
    main()
