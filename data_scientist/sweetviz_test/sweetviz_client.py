#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   sweetviz_client.py
@Time    :   2023-08-22 09:27
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""

import pandas as pd
import sweetviz


if __name__ == '__main__':
    csv_url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
    iris = pd.read_csv(csv_url, header=None)
    report = sweetviz.analyze(iris)
    report.show_html()
