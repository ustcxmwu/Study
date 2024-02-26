#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   line_profiler_client.py
@Time    :   2022/9/29 11:12
@Author  :   Wu Xiaomin <xmwu@mail.ustc.edu.cn>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
"""
from line_profiler import LineProfiler


def findSum(numbers, queries):
    # Write your code here
    res = [0] * len(queries)
    for idx, (i, j, x) in enumerate(queries):
        for xx in numbers[i - 1 : j]:
            if xx == 0:
                res[idx] += x
            else:
                res[idx] += xx
    return res


if __name__ == "__main__":
    lp = LineProfiler()
    func = lp(findSum)
    func([5, 10, 10], [[1, 2, 5]])
    lp.print_stats()
