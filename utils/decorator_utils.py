#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   decorator_utils.py
@Time    :   2024-04-15 13:47
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""

import functools
import time

from decorator import decorator


@decorator
def warn_slow(func, timelimit=60, *args, **kw):
    t0 = time.time()
    result = func(*args, **kw)
    dt = time.time() - t0
    if dt > timelimit:
        print(f"{func.__name__} took {dt} seconds.")
    return result


@decorator
def handle_exceptions(func, p1=1, *args, **kwargs):
    try:
        print(p1)
        return func(*args, **kwargs)
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")


def logit(name):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            value = func(*args, **kwargs)
            print(f'{name} is calling: ' + func.__name__)
            return value

        return wrapper

    return decorator


@logit(name='oldbird')
@handle_exceptions(p1=111)
def add(x, y):
    return x + y


if __name__ == '__main__':
    c = add(1, 2)
    print(c)
