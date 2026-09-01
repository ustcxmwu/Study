#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   tools.py
@Time    :   2026-07-10 09:14
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks, savgol_filter


if __name__ == '__main__':
    # 1. 准备数据
    np.random.seed(10)
    time = np.linspace(0, 50, 500)
    price_vals = (
        100 + np.sin(time) * 10 + np.random.normal(0, 1, len(time)).cumsum()
    )
    df = pd.DataFrame({"Price": price_vals})

    # 2. 方案 A：使用原始数据 + prominence 约束 (显式物理过滤)
    peaks_a, _ = find_peaks(df["Price"].values, distance=20, prominence=5)
    troughs_a, _ = find_peaks(-df["Price"].values, distance=20, prominence=5)

    # 3. 方案 B：先用 Savitzky-Golay 平滑，再找拐点 (信号学过滤)
    df["Smoothed"] = savgol_filter(df["Price"].values, window_length=25, polyorder=3)
    peaks_b, _ = find_peaks(df["Smoothed"].values, distance=20)
    troughs_b, _ = find_peaks(-df["Smoothed"].values, distance=20)

    # --- 可视化对比 ---
    plt.figure(figsize=(14, 6))

    # 子图 1：Prominence 过滤效果
    plt.subplot(1, 2, 1)
    plt.plot(df["Price"], label="Original Price", color="gray", alpha=0.6)
    plt.scatter(
        peaks_a,
        df["Price"].iloc[peaks_a],
        color="red",
        marker="^",
        s=60,
        label="Filtered Peaks (Prominence)",
    )
    plt.scatter(
        troughs_a,
        df["Price"].iloc[troughs_a],
        color="green",
        marker="v",
        s=60,
        label="Filtered Troughs (Prominence)",
    )
    plt.title("Method 1: Prominence Filter")
    plt.legend()

    # 子图 2：Savgol 平滑后效果
    plt.subplot(1, 2, 2)
    plt.plot(df["Price"], label="Original Price", color="gray", alpha=0.3)
    plt.plot(df["Smoothed"], label="Savitzky-Golay Smoothed", color="blue")
    plt.scatter(
        peaks_b,
        df["Smoothed"].iloc[peaks_b],
        color="red",
        marker="^",
        s=60,
        label="Peaks",
    )
    plt.scatter(
        troughs_b,
        df["Smoothed"].iloc[troughs_b],
        color="green",
        marker="v",
        s=60,
        label="Troughs",
    )
    plt.title("Method 2: Signal Smoothing (Savgol)")
    plt.legend()

    plt.tight_layout()
    plt.show()