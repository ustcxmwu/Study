#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   din.py
@Time    :   2026-08-12 16:35
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn
import torch.optim as optim


class LocalActivationUnit(nn.Module):
    """DIN 的核心：局部激活单元（计算候选商品与历史行为的权重）"""

    def __init__(self, emb_dim):
        super(LocalActivationUnit, self).__init__()
        # 输入包括: [query, key, query - key, query * key] -> 维度是 emb_dim * 4
        self.mlp = nn.Sequential(
            nn.Linear(emb_dim * 4, 36),
            nn.ReLU(),
            nn.Linear(36, 1),
            nn.ReLU()
        )

    def forward(self, query, keys):
        """
        query: 候选商品 Embedding, 形状 (Batch_size, emb_dim)
        keys: 用户历史行为序列 Embedding, 形状 (Batch_size, seq_len, emb_dim)
        """
        seq_len = keys.size(1)
        # 将 query 复制 seq_len 份，以便与每个历史行为做对应拼接
        queries = query.unsqueeze(1).expand(-1, seq_len, -1)  # (B, T, E)

        # 显式特征交叉：融合差值和乘积，帮助 MLP 更好捕捉相关性
        interaction = queries * keys
        difference = queries - keys

        # 拼接 [q, k, q-k, q*k]
        att_input = torch.cat([queries, keys, difference, interaction], dim=-1)  # (B, T, E*4)

        # 得到未归一化的注意力得分
        att_weights = self.mlp(att_input)  # (B, T, 1)
        return att_weights.permute(0, 2, 1)  # (B, 1, T)


class SimpleDIN(nn.Module):
    def __init__(self, item_vocab_size, emb_dim=16):
        super(SimpleDIN, self).__init__()

        # 物品 Embedding 层（候选商品和历史商品共享）
        self.item_emb = nn.Embedding(item_vocab_size, emb_dim, padding_idx=0)

        # 注意力计算单元
        self.attention_unit = LocalActivationUnit(emb_dim)

        # 后续的 MLP 预测层
        # 输入维度 = 候选商品(emb_dim) + 注意力加权后的用户兴趣(emb_dim) + 额外稠密特征(2)
        mlp_input_dim = emb_dim * 2 + 2
        self.mlp = nn.Sequential(
            nn.Linear(mlp_input_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )

    def forward(self, candidate_item, history_items, dense_features):
        """
        candidate_item: (B,) 候选商品 ID
        history_items: (B, T) 用户历史点击的商品 ID 序列
        dense_features: (B, 2) 其他稠密特征
        """
        # 1. 查表获取 Embedding
        q_emb = self.item_emb(candidate_item)  # (B, E)
        k_emb = self.item_emb(history_items)  # (B, T, E)

        # 2. 计算注意力权重 (Local Activation)
        att_weights = self.attention_unit(q_emb, k_emb)  # (B, 1, T)

        # 3. 加权求和（Attention Pooling）：得到针对当前候选商品的“用户兴趣向量”
        # (B, 1, T) @ (B, T, E) -> (B, 1, E) -> 挤压降维成 (B, E)
        user_interest = torch.matmul(att_weights, k_emb).squeeze(1)

        # 4. 特征拼接并输入 MLP
        x = torch.cat([q_emb, user_interest, dense_features], dim=1)
        score = self.mlp(x)
        return score


if __name__ == '__main__':
    # --- 1. 超参数与模拟数据 ---
    BATCH_SIZE = 3
    ITEM_VOCAB_SIZE = 1000
    SEQ_LEN = 5  # 假设每个用户的历史行为序列固定长度为 5

    # 候选商品 ID
    candidate_item = torch.tensor([102, 33, 505], dtype=torch.long)
    # 用户历史行为序列 ID（比如用户 A 之前买过 12, 45, 88...）
    history_items = torch.tensor([
        [12, 45, 88, 3, 10],
        [99, 102, 5, 22, 7],
        [500, 301, 12, 99, 33]
    ], dtype=torch.long)
    # 其他稠密特征（例如：用户当天活跃时长、候选商品价格归一化）
    dense_features = torch.tensor([
        [0.5, 0.2],
        [0.9, 0.8],
        [0.1, 0.4]
    ], dtype=torch.float32)

    # 真实标签（是否点击）
    labels = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)

    # --- 2. 初始化模型与优化器 ---
    model = SimpleDIN(item_vocab_size=ITEM_VOCAB_SIZE, emb_dim=16)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    # --- 3. 训练一步 ---
    model.train()
    optimizer.zero_grad()

    preds = model(candidate_item, history_items, dense_features)
    loss = criterion(preds, labels)

    loss.backward()
    optimizer.step()

    print(f"DIN Training Loss: {loss.item():.4f}")
    print(f"DIN Predictions (CTR): {preds.squeeze().detach().numpy()}")