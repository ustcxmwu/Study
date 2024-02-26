#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   scikit_client2.py
@Time    :   2024-01-11 15:31
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer
import pandas as pd
import numpy as np


def cat(x):
    res = []
    for idx, r in x.iterrows():
        res.append(r.jiazu_7 + r.fantuan)
    return np.array([res])


if __name__ == "__main__":
    df = pd.read_csv("./data_240110_v2.csv")
    # df.head()

    c = ColumnTransformer(
        transformers=[("cat", FunctionTransformer(cat), ["jiazu_7", "fantuan"])],
        remainder="passthrough",
    )

    res = c.fit_transform(df)
    res.to_csv("./data_240110_v3.csv", index=False)
