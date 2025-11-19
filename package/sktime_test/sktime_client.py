#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   sktime_client.py
@Time    :   2025-11-14 10:30
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2025, Wu Xiaomin
@Desc    :
"""

from sktime.forecasting.naive import NaiveForecaster
import numpy as np
import pandas as pd


if __name__ == '__main__':

    # 模拟销售数据
    sales_data = pd.Series(np.random.randint(100, 200, 36), index=pd.period_range("2021-01", periods=36, freq="M"))

    forecaster = NaiveForecaster(strategy="last")
    forecaster.fit(sales_data)
    next_6_months = forecaster.predict(fh=[1, 2, 3, 4, 5, 6])
    print(next_6_months)
