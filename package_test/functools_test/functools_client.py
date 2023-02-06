#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from functools import lru_cache
from timeit import default_timer


def factorial(n):
    return n * factorial(n-1) if n else 1


@lru_cache(None)
def fact(n):
    return n * fact(n-1) if n else 1



if __name__ == '__main__':
    s1 = default_timer()
    factorial(100)
    e1 = default_timer()
    print("no cache {}".format(e1 - s1))

    s2 = default_timer()
    fact(100)
    e2 = default_timer()
    print("lru_cache {}".format(e2 - s2))
