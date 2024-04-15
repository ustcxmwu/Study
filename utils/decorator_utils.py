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


def cache(func):
    cached = {}

    def cache_wrapper(*args, **kwargs):
        key = str(args) + str(kwargs)
        if key in cached:
            return cached[key]
        result = func(*args, **kwargs)
        cached[key] = result
        return result

    return cache_wrapper


def retry(max_retries):
    def _decorator(func):
        def _wrapper(*args, **kwargs):
            attempts = 0
            while attempts < max_retries:
                try:
                    return func(*args, **kwargs)
                except Exception:
                    attempts += 1
                    time.sleep(1)
            raise Exception(f"函数 {func.__name__} 执行失败,已达到最大重试次数")

        return _wrapper

    return _decorator


def log_to_file_decorator(log_file_path):
    def _decorator(func):
        def _wrapper(*args, **kwargs):
            with open(log_file_path, "a") as f:
                f.write(f"调用函数 {func.__name__},参数为：args={args},kwargs={kwargs}\n")
            result = func(*args, **kwargs)
            with open(log_file_path, "a") as f:
                f.write(f"函数 {func.__name__} 的返回结果为：{result}\n")
            return result

        return _wrapper

    return _decorator


def logit(name):
    def _decorator(func):
        @functools.wraps(func)
        def _wrapper(*args, **kwargs):
            value = func(*args, **kwargs)
            print(f'{name} is calling: ' + func.__name__)
            return value

        return _wrapper

    return _decorator


@logit(name='oldbird')
@handle_exceptions(p1=111)
def add(x, y):
    return x + y


if __name__ == '__main__':
    c = add(1, 2)
    print(c)
