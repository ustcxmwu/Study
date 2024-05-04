#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   plot_utils.py
@Time    :   2024-05-03 23:58
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import platform
from typing import List

import matplotlib.pyplot as plt
import numpy as np

if "windows" in platform.system().lower():
    plt.rcParams["font.sans-serif"] = ["SimHei"]
elif "darwin" in platform.system().lower():
    plt.rcParams["font.sans-serif"] = ["Arial Unicode MS"]
elif "linux" in platform.system().lower():
    # ubuntu 安装字体参考： https://blog.csdn.net/takedachia/article/details/131017286
    plt.rcParams["font.sans-serif"] = ["simhei"]
else:
    raise ValueError(f"unsupported system {platform.system()}")

plt.rcParams["axes.unicode_minus"] = False


def get_radar_chart(legends: List[str], data_groups: List[List[float]], labels: List[str]):
    # data_groups = [
    #     [3, 5, 2, 6, 8, 4],  # 第一组数据
    #     [1, 2, 4, 3, 7, 9],  # 第二组数据
    #     [0, 1, 3, 2, 4, 6]  # 第三组数据
    # ]
    #
    # labels = ['A', 'B', 'C', 'D', 'E', 'F']
    term_cnt = len(labels) if len(labels) < 12 else 12
    data_groups = [d[:term_cnt] for d in data_groups]
    legends = legends[:term_cnt]
    labels = labels[:term_cnt]
    angles = np.linspace(0, 2 * np.pi, term_cnt, endpoint=False).tolist()

    clist = ['blue', 'red', 'green', 'black', 'darkgreen', 'lime', 'gold', 'purple', 'green', 'cyan', 'salmon', 'grey',
             'mediumvioletred', 'darkkhaki', 'gray', 'darkcyan', 'violet', 'powderblue']

    fig, ax = plt.subplots(figsize=(6, 6), subplot_kw=dict(polar=True))

    for i, data in enumerate(data_groups):
        ax.fill(angles, data, color=clist[i], alpha=0.25)
        ax.plot(angles, data, 'o', label=legends[i], markeredgewidth=2, markeredgecolor=clist[i])
        ax.set_thetagrids(np.degrees(angles), labels)
        plt.grid(True)
        # plt.pause(0.1)  # 暂停一段时间以便于观察每组数据的绘制过程
    plt.legend(loc='best')
    return fig


def rgb_to_hex(r: int, g: int, b: int) -> str:
    return f"#{r:02x}{g:02x}{b:02x}"


if __name__ == '__main__':
    a = 15
    print(f"{a:02x}")
    print(rgb_to_hex(100, 100, 100))
