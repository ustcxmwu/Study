#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   pygwalker_client.py
@Time    :   2023-12-15 15:24
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
import pygwalker as pyg

if __name__ == "__main__":
    df = pd.read_csv("../dataset/titanic.csv")
    walk = pyg.walk(df, env="Streamlit")
