#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   train.py
@Time    :   2024-05-14 21:27
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   深度学习网络训练脚本
"""

import random
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from code_template.dl.dataset import MyDataset
from code_template.dl.model import Model
from code_template.dl.utils import logger, plot_learning_curve

seed = 42
torch.manual_seed(seed)
np.random.seed(seed)
random.seed(seed)


class Args:
    def __init__(self):
        self.batch_size = 32
        self.epochs = 10
        self.lr = 0.01
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def train(model, optimizer, criterion, train_loader, args):
    train_epochs_loss = []
    start_time = time.time()
    for i in range(args.epochs):
        logger.info(f"Epoch num: {i + 1}".center(80, "="))
        model.train()
        train_epoch_loss = []
        for idx, (features, targets) in enumerate(train_loader):
            features = features.to(args.device)
            targets = targets.to(args.device)

            outputs = model(features)
            loss = criterion(outputs, targets)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_epoch_loss.append(loss.item())
            if idx % 100 == 0:
                end_time = time.time()
                logger.info(f"训练时间:{end_time - start_time}")
                logger.info(f"训练次数:{idx}, Loss:{loss.item()}")
        train_epochs_loss.append(np.mean(train_epoch_loss))

        # Optional:
        #   1. 验证并存储在验证集上表现最好的模型
        #   2. 设置early stopping：如果验证集上的表现在超过某个阈值的次数内仍然没有变好，就可以停了
    end_time = time.time()
    logger.info(f"训练结束, 总训练时间:{end_time - start_time}")

    return model, train_epochs_loss


def test(model, test_loader, args):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(args.device)
            targets = targets.to(args.device)

            outputs = model(features)
            test_loss += nn.CrossEntropyLoss()(outputs, targets).item()
            pred = outputs.argmax(dim=1, keepdim=True)
            correct += pred.eq(targets.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)
    accuracy = 100. * correct / len(test_loader.dataset)
    logger.info(
        f"Test set: Average loss: {test_loss:.4f}, Accuracy: {correct}/{len(test_loader.dataset)} ({accuracy:.2f}%)")
    return test_loss, accuracy


def main():
    args = Args()

    train_data = MyDataset()
    train_dataloader = DataLoader(train_data, batch_size=args.batch_size)
    logger.info(f"训练集的长度为{len(train_data)}")

    model = Model()
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    model, train_loss = train(model, optimizer, loss, train_dataloader, args)
    plot_learning_curve(train_loss)

    test_data = MyDataset()
    test_dataloader = DataLoader(test_data, batch_size=args.batch_size)
    logger.info(f"测试集的长度为{len(test_data)}")

    test(model, test_dataloader, args)


if __name__ == '__main__':
    pass
