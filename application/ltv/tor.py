#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   tor.py
@Time    :   2024-09-04 10:46
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


if __name__ == '__main__':
    kl_loss = nn.KLDivLoss(reduction="batchmean")
    # input should be a distribution in the log space
    input = F.log_softmax(torch.randn(3, 5, requires_grad=True), dim=1)
    print(input)
    # Sample a batch of distributions. Usually this would come from the dataset
    target = F.softmax(torch.rand(3, 5), dim=1)
    print(target)
    output = kl_loss(input, target)
    print(output)