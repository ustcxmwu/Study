#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   talib_client.py
@Time    :   2023-12-04 17:39
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import numpy
import talib

if __name__ == "__main__":
    close = numpy.random.random(100)
    output = talib.SMA(close)
    print(output)
