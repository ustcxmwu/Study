#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   dtale_client.py
@Time    :   2023-12-15 11:02
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
import dtale

if __name__ == "__main__":
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris = pd.read_csv(csv_url, header=None)
    dtale.show(iris, subprocess=False)
    # d.open_browser()
