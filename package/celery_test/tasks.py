#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   tasks.py
@Time    :   2023-08-21 13:51
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import time

from app import divide


if __name__ == '__main__':
    task = divide.delay(1, 3)
    while True:
        print(task.state)
        if task.state == "SUCCESS":
            break
        time.sleep(0.1)
