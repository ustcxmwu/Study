#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@File    :   sk_simple_lr_client.py
@Time    :   2022/9/6 16:38
@Author  :   Wu Xiaomin <xmwu@mail.ustc.edu.cn>
@Version :   1.0
@License :   (C)Copyright 2020-2022, Wu Xiaomin
@Desc    :   
"""
import numpy as np
from sklearn import linear_model
import matplotlib.pyplot as plt
import pandas as pd


if __name__ == "__main__":
    x = np.linspace(0, 10, 30)
    y = 2 * x + 3
    print(f"x: {x}")
    print(f"y: {y}")
    plt.scatter(x, y)
    plt.show()

    x = [[i] for i in x]
    y = [[i] for i in y]

    model = linear_model.LinearRegression()
    model.fit(x, y)
    x_ = [[3], [4], [7]]
    y_ = model.predict(x_)
    print("y_: {y_}")

    print(model.coef_)
    print(model.intercept_)
    plt.scatter(x, y)
    plt.plot(x_, y_, color="red", linewidth=3.0, linestyle="-")
    plt.show()
