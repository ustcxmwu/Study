#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   tansformer.py
@Time    :   2023-12-12 17:40
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""


if __name__ == "__main__":
    import torch
    import torch.nn.functional as F

    # Query, Key
    query = torch.tensor([1, 0.5], dtype=torch.float)
    key = torch.tensor([[1, 0], [0, 1]], dtype=torch.float)

    # 相似度计算
    similarity = query.matmul(key)
    print(similarity)

    # 权重分配
    weights = F.softmax(similarity, dim=-1)
    # 输出：tensor([0.7311, 0.2689])
    print(weights)

    value = torch.tensor([[1, 2], [3, 4]], dtype=torch.float)
    output = weights.matmul(value)
    print(output)
