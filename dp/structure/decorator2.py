import functools


def memoize(fn):
    known = dict()

    @functools.wraps(fn)
    def memoizer(*args):
        if args not in known:
            known[args] = fn(*args)
        return known[args]

    return memoizer


@memoize
def nsum(n):
    assert (n>=0), "n must be >= 0"
    return 0 if n== 0 else n + nsum(n-1)


@memoize
def fibonacci(n):
    assert (n>=0), "n must be >= 0"
    return n if n in (0, 1) else fibonacci(n-1) + fibonacci(n-2)


def main():
    from timeit import Timer
    measure = [
        {"exec": "fibonacci(100)", "import": "fibonacci", "func": fibonacci},
        {"exec": "nsum(200)", "import": "nsum", "func": nsum}
    ]
    for m in measure:
        t = Timer(m["exec"], "from __main__ import {}".format(m["import"]))
        print("name:{}, executing:{}, time:{}".format(m["func"].__name__, m["exec"], t.timeit()))


if __name__ == "__main__":
    main()

