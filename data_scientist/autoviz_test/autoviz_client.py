#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   autoviz_client.py
@Time    :   2023-12-15 11:40
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
from autoviz.AutoViz_Class import AutoViz_Class


if __name__ == "__main__":
    csv_url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
    iris = pd.read_csv(csv_url, header=None)
    iris.to_csv("./iris.csv")
    # chart_format: html, server, png, jpg, svg, bokeh
    autoviz = AutoViz_Class().AutoViz("./iris.csv", chart_format="html")
