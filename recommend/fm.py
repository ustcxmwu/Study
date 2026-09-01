#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   fm.py
@Time    :   2026-08-12 16:39
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn
import torch.optim as optim


class SparseFM(nn.Module):
    def __init__(self, vocab_size, emb_dim=8):
        super(SparseFM, self).__init__()
        # 一阶特征权重 (Lookup Table)
        self.bias = nn.Parameter(torch.zeros(1))
        self.first_order = nn.Embedding(vocab_size, 1)

        # 二阶隐向量矩阵 V
        self.embeddings = nn.Embedding(vocab_size, emb_dim)
        nn.init.normal_(self.embeddings.weight, mean=0.0, std=0.01)

    def forward(self, x):
        """
        x: 形状为 (Batch_size, num_fields) 的特征索引矩阵
        """
        # 1. 一阶线性求和: sum(w_i * x_i)
        # 形状: (B, F, 1) -> 降维求和 -> (B, 1)
        first_sum = self.first_order(x).sum(dim=1)

        # 2. 二阶交叉项的线性时间优化公式：
        # 0.5 * sum_f [ (sum(v_if * x_i))^2 - sum((v_if * x_i)^2) ]

        # 获取隐向量: (B, F, K)
        v_x = self.embeddings(x)

        # 第一项: (sum(v_if))^2 -> 沿着特征维度(dim=1)求和，然后平方
        sum_pref = v_x.sum(dim=1)  # (B, K)
        sum_pref_sq = sum_pref ** 2  # (B, K)

        # 第二项: sum(v_if^2) -> 平方后沿着特征维度(dim=1)求和
        sq_pref = v_x ** 2  # (B, F, K)
        sq_sum_pref = sq_pref.sum(dim=1)  # (B, K)

        # 二阶交叉结果: 0.5 * sum(sum_pref_sq - sq_sum_pref, dim=1)
        second_order = 0.5 * torch.sum(sum_pref_sq - sq_sum_pref, dim=1, keepdim=True)  # (B, 1)

        # 3. 总体求和并通过 Sigmoid 输出概率
        logits = self.bias + first_sum + second_order
        return torch.sigmoid(logits)


if __name__ == '__main__':
    # --- 1. 超参数与模拟数据 ---
    BATCH_SIZE = 3
    VOCAB_SIZE = 1000  # 特征总词表大小（把所有 User、Item、Tag 映射到一个全局空间或分段）
    EMB_DIM = 8

    # 假设每个样本有 3 个激活的特征索引（例如：[User_ID, Item_ID, Category_ID]）
    x_indices = torch.tensor([
        [12, 452, 33],
        [99, 102, 5],
        [500, 301, 12]
    ], dtype=torch.long)

    # 真实标签（点击 = 1，不点击 = 0）
    labels = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

    # --- 2. 初始化模型与优化器 ---
    model = SparseFM(vocab_size=VOCAB_SIZE, emb_dim=EMB_DIM)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- 3. 训练一步 ---
    model.train()
    optimizer.zero_grad()

    preds = model(x_indices)
    loss = criterion(preds, labels)

    loss.backward()
    optimizer.step()

    print(f"FM Training Loss: {loss.item():.4f}")
    print(f"FM Predictions (CTR): {preds.squeeze().detach().numpy()}")