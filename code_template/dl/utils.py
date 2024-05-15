#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   utils.py
@Time    :   2024-05-14 22:29
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   深度学习 Pytorch 工具函数
"""

import logging
import platform
import time
from pathlib import Path

import matplotlib.pyplot as plt

if "windows" in platform.system().lower():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
elif "darwin" in platform.system().lower():
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
elif "linux" in platform.system().lower():
    # ubuntu 安装字体参考： https://blog.csdn.net/takedachia/article/details/131017286
    plt.rcParams["font.sans-serif"] = ["simhei"]

plt.rcParams["axes.unicode_minus"] = False

LOG_DIR = Path("./logs")
if not LOG_DIR.exists():
    LOG_DIR.mkdir()

logger = logging.getLogger()
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
fh = logging.FileHandler(filename=f"{str(LOG_DIR)}/log_{time.strftime('%Y%m%d')}.log")

formatter = logging.Formatter(
    "%(asctime)s - %(module)s - %(funcName)s - line:%(lineno)d - %(levelname)s - %(message)s"
)
formatter2 = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
fh.setFormatter(formatter)
ch.setFormatter(formatter2)
logger.addHandler(ch)
logger.addHandler(fh)


def plot_learning_curve(loss_record, title=''):
    plt.figure(figsize=(10, 6))
    plt.plot(list(range(len(loss_record))), loss_record)

    plt.xlabel('Training steps')
    plt.ylabel('MSE loss')
    plt.title(f'Learning curve of {title}')
    plt.legend()
    plt.show()
