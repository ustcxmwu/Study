#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   ydata_profile_test.py
@Time    :   2023-12-15 11:23
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
from ydata_profiling import ProfileReport


if __name__ == "__main__":
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris = pd.read_csv(csv_url, header=None)

    profile = ProfileReport(iris, explorative=True)
    profile.to_file("output.html")
