#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   pygwalker_client.py
@Time    :   2023-12-15 13:51
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
import pygwalker as pyg

if __name__ == "__main__":
    df = pd.read_csv("../dataset/titanic.csv")
    walker = pyg.walk(df, return_html=True)
    walker.to_html()
