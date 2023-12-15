#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   klib_client.py
@Time    :   2023-12-15 11:54
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import klib
import pandas as pd

# Klib 分析不成功, Github Star 数过小, 放弃

if __name__ == "__main__":
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris = pd.read_csv(csv_url, header=None)
    # klib.missingval_plot(iris)
    klib.corr_plot(iris, annot=False)
    klib.cat_plot(iris)
