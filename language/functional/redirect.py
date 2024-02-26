#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   redirect.py
@Time    :   2023-12-28 10:32
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import contextlib
import sys
import time

from io import StringIO


@contextlib.contextmanager
def stdout_redirect(where):
    sys.stdout = where
    try:
        yield where
    finally:
        sys.stdout = sys.__stdout__


def foo():
    print("bar")
    time.sleep(3)
    print("bar2")


if __name__ == "__main__":
    with stdout_redirect(StringIO()) as new_stdout:
        foo()
    new_stdout.seek(0)
    print("data from new_stdout:", new_stdout.read())
    new_stdout1 = StringIO()
    with stdout_redirect(new_stdout1):
        foo()
    new_stdout1.seek(0)
    print("data from new_stdout1:", new_stdout1.read())
    # Now with a file object:
    with open("a.txt", mode="w") as f:
        with stdout_redirect(f):
            foo()
    # Just to prove that we actually did put stdout back as we were supposed to
    print("Now calling foo without context")
    foo()
