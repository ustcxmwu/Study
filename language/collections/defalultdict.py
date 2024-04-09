#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   defalultdict.py
@Time    :   2024-04-02 14:06
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
from collections import defaultdict, Counter


if __name__ == '__main__':
    a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    b = defaultdict(Counter)
    for i in a:
        b[i] += 1
    print(b)
