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
from pathlib import Path

import joblib
import numpy as np


if __name__ == '__main__':
    # 加载向量文件：{技能id: 16 维向量}
    embeddings = joblib.load(Path(__file__).with_name("embeddings_joblib.pkl"))

    vectors = np.stack(list(embeddings.values()))
    print(f"Loaded embeddings with shape: {vectors.shape}")

    # 打印前5个向量的示例
    for skill_id, vector in list(embeddings.items())[:5]:
        print(f"Embedding {skill_id}: {vector}")
