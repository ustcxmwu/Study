#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   embeddings_review.py
@Time    :   2026-06-24 11:08
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import joblib


if __name__ == '__main__':
    # 加载向量文件
    embeddings = joblib.load("./embeddings_joblib.pkl")
    print(f"Loaded embeddings with shape: {embeddings.shape}")

    # 打印前5个向量的示例
    for i in range(min(5, len(embeddings))):
        print(f"Embedding {i}: {embeddings[i]}")