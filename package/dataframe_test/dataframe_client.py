#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   dataframe_client.py
@Time    :   2024-05-14 09:12
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import pandas as pd
import numpy as np

def main():
    data = [
        [10, 20, 30],
        [10, 50, 60],
        [70, 80, 90],
        [20, 30, 10],
        [20, 10, 20]
    ]
    df = pd.DataFrame(np.array(data), columns=[f"column_{i}" for i in range(len(data[0]))])

    print(df)
    df["column_4"] = ~df['column_0'].duplicated()
    print(df)


if __name__ == '__main__':
    main()
