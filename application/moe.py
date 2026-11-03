#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   moe.py
@Time    :   2026-09-02 15:55
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class Expert(nn.Module):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU(),
            nn.Linear(d_ff, d_model)
        )

    def forward(self, x):
        return self.net(x)


class SimpleMoE(nn.Module):
    def __init__(self, d_model, d_ff, num_experts, top_k=2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        self.router = nn.Linear(d_model, num_experts)
        self.experts = nn.ModuleList([Expert(d_model, d_ff) for _ in range(num_experts)])

    def forward(self, x):
        # x shape: [batch_size, seq_len, d_model]
        batch_size, seq_len, d_model = x.shape
        x_flat = x.view(-1, d_model)  # [N, d_model]

        # 1. 计算路由概率
        router_logits = self.router(x_flat)  # [N, num_experts]
        router_weights = F.softmax(router_logits, dim=-1)

        # 2. 选择 Top-k 专家
        top_k_weights, top_k_indices = torch.topk(router_weights, self.top_k, dim=-1)  # [N, top_k]
        top_k_weights = top_k_weights / top_k_weights.sum(dim=-1, keepdim=True)  # 归一化

        # 3. 专家计算与结果组合
        final_output = torch.zeros_like(x_flat)
        for i, expert in enumerate(self.experts):
            mask = (top_k_indices == i)  # 寻找分配给第 i 个专家的 token
            if not mask.any():
                continue

            token_idx, k_idx = torch.where(mask)
            if len(token_idx) == 0:
                continue

            expert_input = x_flat[token_idx]
            expert_output = expert(expert_input)

            # 加权累加
            weights = top_k_weights[token_idx, k_idx].unsqueeze(1)
            final_output.index_add_(0, token_idx, expert_output * weights)

        return final_output.view(batch_size, seq_len, d_model)


if __name__ == '__main__':
    # 验证测试
    d_model, d_ff, num_experts = 64, 256, 4
    moe_layer = SimpleMoE(d_model, d_ff, num_experts, top_k=2)
    x = torch.randn(2, 8, d_model)
    output = moe_layer(x)
    print(output.shape)  # 输出: torch.Size([2, 8, 64])
