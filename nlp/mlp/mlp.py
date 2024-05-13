#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   mlp.py
@Time    :   2024-05-13 14:10
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import torch.nn as nn
import torch.nn.functional as F


class MultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        """Initialize the model

        Args:
            input_dim ():
            hidden_dim ():
            output_dim ():
        """
        super(MultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x_in, apply_softmax=False):
        """

        Args:
            x_in (torch.Tensor): an input tensor, x_in.shape should be (batch, input_dim)
            apply_softmax (bool): should be false if used with the cross-entropy losses.

        Returns:
            the resulting tensor. tensor.shape should be (batch, output_dim)
        """
        intermediate = F.relu(self.fc1(x_in))
        output = self.fc2(intermediate)
        if apply_softmax:
            output = F.softmax(output, dim=1)
        return output
