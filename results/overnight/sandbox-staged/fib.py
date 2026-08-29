import argparse

from fib_module import fib


def main():
    parser = argparse.ArgumentParser(description="Compute a Fibonacci number")
    parser.add_argument("n", type=int, help="index of the Fibonacci number")
    args = parser.parse_args()
    print(fib(args.n))


if __name__ == "__main__":
    main()
