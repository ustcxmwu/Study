#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   peaks.py
@Time    :   2026-07-10 09:21
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

# ==========================================
# 1. 构造测试数据：价格从 10 暴涨到 150
# ==========================================
np.random.seed(42)
n_points = 600
time = np.linspace(0, 30, n_points)

# 基础趋势线：从 10 涨到 150 的指数级增长
trend = 10 * np.exp(time * 0.09)

# 制造周期性波动（等比例：在当前价格基础上波动 8%）
cycles = trend * 0.08 * np.sin(time * 2)

# 制造噪声（等比例：当前价格的 1.5%）
noise = trend * 0.015 * np.random.randn(n_points)

prices = trend + cycles + noise
df = pd.DataFrame({"Price": prices})

# ==========================================
# 2. 方案应用与寻峰
# ==========================================

# --- 方案一：对数变换 ---
log_prices = np.log(df["Price"].values)
# 设定对数空间的距离（0.05 约等于 5% 涨跌幅门槛）
peaks_log, _ = find_peaks(log_prices, distance=15, prominence=0.05)
troughs_log, _ = find_peaks(-log_prices, distance=15, prominence=0.05)

# --- 方案二：动态均价比例 ---
rolling_mean = df["Price"].rolling(window=30, min_periods=1).mean().values
# 门槛设定为当前均价的 5%
prom_mean = rolling_mean * 0.05
peaks_mean, _ = find_peaks(df["Price"].values, distance=15, prominence=prom_mean)
troughs_mean, _ = find_peaks(
    -df["Price"].values, distance=15, prominence=prom_mean
)

# --- 方案三：动态波动率 (标准差) ---
rolling_std = df["Price"].rolling(window=30, min_periods=1).std()
# 填充初始点的 NaN，门槛设定为 1.5 倍标准差
prom_std = (rolling_std * 1.5).bfill().values
peaks_std, _ = find_peaks(df["Price"].values, distance=15, prominence=prom_std)
troughs_std, _ = find_peaks(-df["Price"].values, distance=15, prominence=prom_std)


# ==========================================
# 3. 画图比较
# ==========================================
fig, axes = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

# 提取公用的绘图逻辑
def plot_results(ax, title, p_idx, t_idx):
    ax.plot(df["Price"], color="#A0A0A0", alpha=0.6, label="Stock Price")
    ax.scatter(
        p_idx,
        df["Price"].iloc[p_idx],
        color="#D62728",
        marker="^",
        s=50,
        label="Peaks",
    )
    ax.scatter(
        t_idx,
        df["Price"].iloc[t_idx],
        color="#2CA02C",
        marker="v",
        s=50,
        label="Troughs",
    )
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.grid(True, linestyle="--", alpha=0.5)


plot_results(axes[0], "Method 1: Log Transform (Constant % Space)", peaks_log, troughs_log)
plot_results(axes[1], "Method 2: Dynamic Moving Average % Threshold", peaks_mean, troughs_mean)
plot_results(axes[2], "Method 3: Dynamic Volatility (Std/ATR) Threshold", peaks_std, troughs_std)

plt.xlabel("Ticks / Time", fontsize=10)
plt.tight_layout()
plt.show()