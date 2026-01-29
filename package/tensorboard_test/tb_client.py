#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   tb_client.py
@Time    :   2025-11-26 14:35
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2025, Wu Xiaomin
@Desc    :
"""

import torch
from torch.utils.tensorboard import SummaryWriter

if __name__ == "__main__":
    # 假设的训练参数
    num_epochs = 10
    learning_rate = 0.01

    # 1. 创建 SummaryWriter 实例
    # logs 会被写入 'runs/multi_level_example' 目录
    writer = SummaryWriter("./runs/multi_level_example2")

    # 假设的模型（此处仅用于模拟计算梯度）
    model = torch.nn.Linear(10, 2)

    print("开始模拟训练过程...")

    for epoch in range(1, num_epochs + 1):
        # 模拟指标数据 (实际中这些数据来自您的模型输出)
        train_loss = 1.0 / (epoch * 0.5 + 1)
        val_loss = 1.0 / (epoch * 0.4 + 1)
        train_acc = 0.5 + epoch * 0.05
        val_acc = 0.5 + epoch * 0.04

        # 模拟梯度计算（以计算梯度范数）
        loss = torch.tensor(train_loss, requires_grad=True)
        loss.backward()

        # 假设编码器和分类器的梯度范数
        # 在实际中，您需要遍历 model.named_parameters() 来分别计算
        encoder_grad_norm = torch.rand(1).item() + 0.5 / epoch
        classifier_grad_norm = torch.rand(1).item() + 0.3 / epoch

        # --- 使用不同层级的标签记录指标 ---

        # 1. 两层标签：指标类别 / 数据集
        writer.add_scalar("Loss/Train", train_loss, epoch)
        writer.add_scalar("Loss/Validation", val_loss, epoch)
        writer.add_scalar("Accuracy/Train", train_acc, epoch)
        writer.add_scalar("Accuracy/Validation", val_acc, epoch)

        # 2. 三层标签：模块 / 指标类别 / 数据集/类型
        writer.add_scalar("Gradients/Norm/Encoder", encoder_grad_norm, epoch)
        writer.add_scalar("Gradients/ABC/Encoder", encoder_grad_norm, epoch)
        writer.add_scalar("Gradients/Norm/Classifier", classifier_grad_norm, epoch)
        writer.add_scalar("Gradients/ABC/Classifier", classifier_grad_norm, epoch)

        print(f"Epoch {epoch}: Train Loss={train_loss:.4f}, Val Acc={val_acc:.4f}")

    writer.close()
    print("\n数据写入完成。请运行 TensorBoard 命令查看：")
    print("tensorboard --logdir=runs/multi_level_example")
