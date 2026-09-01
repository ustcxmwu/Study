#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   dnn.py
@Time    :   2026-08-12 16:33
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn
import torch.optim as optim


class SimpleRecommenderDNN(nn.Module):
    def __init__(self, user_vocab_size, item_vocab_size, cat_vocab_size,
                 emb_dim=16, dense_dim=3, hidden_dims=[64, 32]):
        super(SimpleRecommenderDNN, self).__init__()

        # 1. Embedding 层：将高维稀疏 ID 映射为低维稠密向量
        self.user_emb = nn.Embedding(user_vocab_size, emb_dim)
        self.item_emb = nn.Embedding(item_vocab_size, emb_dim)
        self.cat_emb = nn.Embedding(cat_vocab_size, emb_dim)

        # 2. 计算拼接后的特征总维度
        # 3个 Embedding 特征 (emb_dim * 3) + 稠密特征数量 (dense_dim)
        input_dim = emb_dim * 3 + dense_dim

        # 3. 多层感知机 (MLP) 特征交叉层
        layers = []
        for h_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, h_dim))
            layers.append(nn.ReLU())
            layers.append(nn.BatchNorm1d(h_dim))
            layers.append(nn.Dropout(0.2))
            input_dim = h_dim

        self.mlp = nn.Sequential(*layers)

        # 4. 输出层：输出点击概率 (0 ~ 1)
        self.output_layer = nn.Linear(hidden_dims[-1], 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, user_id, item_id, cat_id, dense_features):
        # 查表获取 Embedding
        u_emb = self.user_emb(user_id)
        i_emb = self.item_emb(item_id)
        c_emb = self.cat_emb(cat_id)

        # 拼接所有特征 (Batch_size, total_feature_dim)
        x = torch.cat([u_emb, i_emb, c_emb, dense_features], dim=1)

        # 通过 MLP 进行非线性特征交叉
        x = self.mlp(x)

        # 输出预测得分
        logits = self.output_layer(x)
        return self.sigmoid(logits)


if __name__ == '__main__':
    # --- 1. 超参数配置 ---
    BATCH_SIZE = 4
    USER_VOCAB_SIZE = 1000  # 用户总数
    ITEM_VOCAB_SIZE = 5000  # 物品总数
    CAT_VOCAB_SIZE = 50  # 类别总数
    DENSE_DIM = 3  # 稠密特征数

    # --- 2. 准备模拟批次数据 (Batch Data) ---
    user_ids = torch.tensor([12, 45, 88, 3], dtype=torch.long)
    item_ids = torch.tensor([102, 33, 4051, 99], dtype=torch.long)
    cat_ids = torch.tensor([5, 2, 12, 1], dtype=torch.long)
    # 稠密特征：例如 [用户年龄归一化, 历史CTR, 物品平均分]
    dense_features = torch.tensor([
        [0.2, 0.55, 0.8],
        [0.8, 0.12, 0.4],
        [0.4, 0.90, 0.9],
        [0.5, 0.30, 0.5]
    ], dtype=torch.float32)

    # 真实标签：是否点击 (0 或 1)
    labels = torch.tensor([[1.0], [0.0], [1.0], [0.0]], dtype=torch.float32)

    # --- 3. 初始化模型与优化器 ---
    model = SimpleRecommenderDNN(
        user_vocab_size=USER_VOCAB_SIZE,
        item_vocab_size=ITEM_VOCAB_SIZE,
        cat_vocab_size=CAT_VOCAB_SIZE
    )

    criterion = nn.BCELoss()  # 二分类交叉熵损失
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 4. 训练步 (Training Step) ---
    model.train()
    optimizer.zero_grad()

    # 前向传播：预测点击概率
    predictions = model(user_ids, item_ids, cat_ids, dense_features)

    # 计算损失
    loss = criterion(predictions, labels)

    # 反向传播与优化
    loss.backward()
    optimizer.step()

    print(f"Training Loss: {loss.item():.4f}")
    print(f"Predictions: {predictions.squeeze().detach().numpy()}")