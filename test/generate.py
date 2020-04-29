

def fib(n):
    a, b = 1, 1
    while a < n:
        yield a
        a, b = b, a+b


if __name__ == '__main__':
    for i, f in enumerate(fib(10)):
        print(f)

    print(fib(2))
