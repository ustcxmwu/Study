#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   find_peak.py
@Time    :   2026-07-10 09:32
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import numpy as np
import pandas as pd
from scipy.signal import find_peaks


def calculate_log_extrema(
    df: pd.DataFrame,
    price_col: str = "close",
    distance: int = 15,
    prominence_pct: float = 0.05,
) -> list:
    """使用对数空间变换计算股票价格的局部波峰与波谷值（优化索引版）。

    参数:
    ----------
    df : pd.DataFrame
        包含股票价格的 DataFrame，其 index 即为时间（如 DatetimeIndex 或 RangeIndex）。
    price_col : str, 默认 'close'
        用于计算峰谷的价格列名。
    distance : int, 默认 15
        相邻同类拐点之间的最小样本距离。
    prominence_pct : float, 默认 0.05 (即 5%)
        波峰/波谷相对于周围环境的最小突起/凹陷比例。

    返回:
    -------
    list of dict
        每个元素为一个 dict，格式如: {'time': df.index中的时间类型, 'price': float, 'type': 'peak'/'trough'}
    """
    if price_col not in df.columns:
        raise ValueError(f"DataFrame 中未找到价格列: {price_col}")

    prices = df[price_col].values

    # 1. 对数变换以消除绝对价格影响
    log_prices = np.log(prices)

    # 2. 在对数空间寻峰与寻谷
    peaks_idx, _ = find_peaks(
        log_prices, distance=distance, prominence=prominence_pct
    )
    troughs_idx, _ = find_peaks(
        -log_prices, distance=distance, prominence=prominence_pct
    )

    # 3. 提取 DataFrame 原始索引及价格
    results = []

    # 直接使用 df.index[idx] 获取，完美保留原索引的所有时间属性（如 tz, freq 等）
    for idx in peaks_idx:
        results.append(
            {
                "time": df.index[idx],
                "price": float(prices[idx]),
                "type": "peak",
                "_sort_key": idx,  # 引入位置索引作为排序键，性能更高
            }
        )

    for idx in troughs_idx:
        results.append(
            {
                "time": df.index[idx],
                "price": float(prices[idx]),
                "type": "trough",
                "_sort_key": idx,
            }
        )

    # 4. 显式按时间先后顺序（物理位置）排序，并移除临时排序键
    results.sort(key=lambda x: x["_sort_key"])

    for item in results:
        item.pop("_sort_key")

    return results