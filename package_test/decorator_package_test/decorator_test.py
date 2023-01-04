#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   decorator_test.py
@Time    :   2023-01-03 17:57
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import threading
import time

from decorator import decorator


@decorator
def trace(f, *args, **kw):
    kwstr = ', '.join('%r: %r' % (k, kw[k]) for k in sorted(kw))
    print("calling %s with args %s, {%s}" % (f.__name__, args, kwstr))
    return f(*args, **kw)


@trace
def func():
    pass


@decorator
def blocking(f, msg='blocking', *args, **kw):
    if not hasattr(f, "thread"):  # no thread running
        def set_result():
            f.result = f(*args, **kw)

        f.thread = threading.Thread(None, set_result)
        f.thread.start()
        return msg
    elif f.thread.is_alive():
        return msg
    else:  # the thread is ended, return the stored result
        del f.thread
        return f.result


@blocking(msg="Please wait ...")
def read_data():
    time.sleep(3)  # simulate a blocking resource
    return "some data"


if __name__ == '__main__':
    # func()
    print(read_data())
