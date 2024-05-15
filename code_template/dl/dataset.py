#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   dataset.py
@Time    :   2024-05-14 21:29
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   深度学习自定义数据集
"""
import os

from torch.utils.data import Dataset


class MyDataset(Dataset):
    """
    自定义数据集
    """
    def __init__(self):
        """
        可以在这个部分根据mode（train/val/test）入参来对数据集进行划分
        """
        pass

    def __getitem__(self, index):
        """
        获取数据集中的一个样本, 应该返回一个 Tensor
        Args:
            index (int):

        Returns:
            sample (torch.Tensor): 一个样本
            target (torch.Tensor): 样本对应的标签
        """
        pass

    def __len__(self):
        pass

    @classmethod
    def from_csv(cls, filename: os.PathLike):
        pass


if __name__ == '__main__':
    pass
