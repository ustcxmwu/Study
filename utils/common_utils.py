#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   common_utils.py
@Time    :   2024-04-09 15:36
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""


def flatten_list_tuple_range(l):
    for el in l:
        if isinstance(el, (list, tuple, range)):
            yield from flatten_list_tuple_range(el)
        else:
            yield el
