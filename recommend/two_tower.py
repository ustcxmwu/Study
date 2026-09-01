#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   two_tower.py
@Time    :   2026-08-12 16:47
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


class UserTower(nn.Module):
    """用户塔：负责把用户特征压缩为一个低维稠密向量"""

    def __init__(self, user_vocab_size, emb_dim=16, dense_dim=2, hidden_dim=32, output_dim=16):
        super(UserTower, self).__init__()
        self.user_emb = nn.Embedding(user_vocab_size, emb_dim)

        # 输入维度 = 用户 Embedding 维度 + 用户稠密特征维度
        input_dim = emb_dim + dense_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 最终输出用户向量
        )

    def forward(self, user_ids, user_dense):
        u_emb = self.user_emb(user_ids)
        x = torch.cat([u_emb, user_dense], dim=1)
        user_vector = self.mlp(x)
        # 工业界常用 L2 归一化，方便后面用余弦相似度检索
        return F.normalize(user_vector, p=2, dim=1)


class ItemTower(nn.Module):
    """物品塔：负责把物品特征压缩为一个与用户向量同维度的低维稠密向量"""

    def __init__(self, item_vocab_size, cat_vocab_size, emb_dim=16, dense_dim=2, hidden_dim=32, output_dim=16):
        super(ItemTower, self).__init__()
        self.item_emb = nn.Embedding(item_vocab_size, emb_dim)
        self.cat_emb = nn.Embedding(cat_vocab_size, emb_dim)

        # 输入维度 = 物品 Embedding(emb_dim*2) + 物品稠密特征维度
        input_dim = emb_dim * 2 + dense_dim

        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)  # 最终输出物品向量
        )

    def forward(self, item_ids, cat_ids, item_dense):
        i_emb = self.item_emb(item_ids)
        c_emb = self.cat_emb(cat_ids)
        x = torch.cat([i_emb, c_emb, item_dense], dim=1)
        item_vector = self.mlp(x)
        # 同样进行 L2 归一化
        return F.normalize(item_vector, p=2, dim=1)


class TwoTowerModel(nn.Module):
    """双塔模型总体结构"""

    def __init__(self, user_tower, item_tower):
        super(TwoTowerModel, self).__init__()
        self.user_tower = user_tower
        self.item_tower = item_tower

    def forward(self, user_ids, user_dense, item_ids, cat_ids, item_dense):
        # 1. 独立计算用户向量
        u_vec = self.user_tower(user_ids, user_dense)  # (B, Output_Dim)

        # 2. 独立计算物品向量
        i_vec = self.item_tower(item_ids, cat_ids, item_dense)  # (B, Output_Dim)

        # 3. 计算双塔向量的内积（余弦相似度，因为前面做了归一化）
        # (B, 1) -> 逐元素相乘后按行求和
        score = torch.sum(u_vec * i_vec, dim=1, keepdim=True)

        # 将内积通过 Sigmoid 映射到 0~1 之间作为预测概率
        return torch.sigmoid(score), u_vec, i_vec


if __name__ == '__main__':
    # --- 1. 超参数配置 ---
    BATCH_SIZE = 3
    USER_VOCAB_SIZE = 100
    ITEM_VOCAB_SIZE = 500
    CAT_VOCAB_SIZE = 20
    EMB_DIM = 16
    OUTPUT_DIM = 8  # 双塔最终输出向量的维度

    # --- 2. 模拟批次数据 ---
    user_ids = torch.tensor([12, 45, 8], dtype=torch.long)
    user_dense = torch.tensor([[0.5, 0.2], [0.9, 0.1], [0.3, 0.8]], dtype=torch.float32)

    item_ids = torch.tensor([102, 33, 405], dtype=torch.long)
    cat_ids = torch.tensor([5, 2, 12], dtype=torch.long)
    item_dense = torch.tensor([[0.1, 0.9], [0.4, 0.5], [0.8, 0.1]], dtype=torch.float32)

    labels = torch.tensor([[1.0], [0.0], [1.0]], dtype=torch.float32)  # 是否点击

    # --- 3. 初始化模块与模型 ---
    u_tower = UserTower(USER_VOCAB_SIZE, emb_dim=EMB_DIM, dense_dim=2, output_dim=OUTPUT_DIM)
    i_tower = ItemTower(ITEM_VOCAB_SIZE, CAT_VOCAB_SIZE, emb_dim=EMB_DIM, dense_dim=2, output_dim=OUTPUT_DIM)
    model = TwoTowerModel(u_tower, i_tower)

    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)

    # --- 4. 训练一步 ---
    model.train()
    optimizer.zero_grad()

    preds, user_vecs, item_vecs = model(user_ids, user_dense, item_ids, cat_ids, item_dense)
    loss = criterion(preds, labels)

    loss.backward()
    optimizer.step()

    print(f"Two-Tower Training Loss: {loss.item():.4f}")
    print(f"Predictions: {preds.squeeze().detach().numpy()}")
    print(f"User Vector Shape (可存入线上缓存): {user_vecs.shape}")
    print(f"Item Vector Shape (可存入向量检索库): {item_vecs.shape}")
