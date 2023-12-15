#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   dabl_client.py
@Time    :   2023-12-15 12:04
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
import dabl

# 画图未成功, Github Star 数量过少,不考虑了
if __name__ == "__main__":
    df = pd.read_csv("../dataset/titanic.csv")
    df_clean = dabl.clean(df, verbose=1)
    dabl.plot(df_clean, target_col="Age")
