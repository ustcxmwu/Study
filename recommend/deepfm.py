#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   deepfm.py
@Time    :   2026-08-12 16:44
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn


class DeepFM(nn.Module):
    def __init__(self, vocab_size, emb_dim, num_fields, hidden_dims=[32, 16]):
        super(DeepFM, self).__init__()

        # 1. 共享 Embedding 层
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        # 一阶线性项权重 (每个特征一个权重)
        self.first_order_weight = nn.Embedding(vocab_size, 1)

        # 2. DNN 模块 (高阶特征交叉)
        # 输入维度 = 特征数量 * 向量维度
        input_dim = num_fields * emb_dim
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            input_dim = h_dim
        self.dnn = nn.Sequential(*layers)
        self.dnn_final = nn.Linear(hidden_dims[-1], 1)

        # 全局偏置
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        """x: (Batch_size, num_fields)"""

        # A. FM 部分
        # 一阶项: sum(w_i * x_i)
        first_order = self.first_order_weight(x).sum(dim=1)

        # 二阶项: 0.5 * sum((sum_v)^2 - sum(v^2))
        v = self.embedding(x)  # (B, F, E)
        sum_v = v.sum(dim=1)  # (B, E)
        sum_v_sq = sum_v ** 2
        v_sq = v ** 2
        sum_v_sq_sum = v_sq.sum(dim=1)
        second_order = 0.5 * (sum_v_sq - sum_v_sq_sum).sum(dim=1, keepdim=True)
        fm_out = first_order + second_order

        # B. DNN 部分
        # 展平 Embedding (Batch, F*E)
        dnn_in = v.view(v.size(0), -1)
        dnn_out = self.dnn_final(self.dnn(dnn_in))

        # C. 最终融合
        return torch.sigmoid(self.bias + fm_out + dnn_out)

if __name__ == '__main__':
    # --- 配置 ---
    BATCH_SIZE = 2
    NUM_FIELDS = 3  # 假设每个样本有 3 个特征（User, Item, Category）
    VOCAB_SIZE = 100  # 总特征空间大小
    EMB_DIM = 4  # 每个特征嵌入为 4 维

    # --- 模拟数据 ---
    # 每行是一个样本的 3 个特征 ID
    input_data = torch.tensor([
        [10, 20, 5],
        [1, 33, 9]
    ], dtype=torch.long)
    labels = torch.tensor([[1.0], [0.0]], dtype=torch.float32)

    # --- 模型 ---
    model = DeepFM(VOCAB_SIZE, EMB_DIM, NUM_FIELDS)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.BCELoss()

    # --- 训练逻辑 ---
    model.train()
    optimizer.zero_grad()

    # 前向传播
    preds = model(input_data)
    loss = criterion(preds, labels)

    # 反向传播 (此时 FM 和 DNN 同时更新 embedding 层的参数)
    loss.backward()
    optimizer.step()

    print(f"DeepFM Training Loss: {loss.item():.4f}")
    print(f"预测概率: {preds.squeeze().detach().numpy()}")
