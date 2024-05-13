#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   elman_rnn.py
@Time    :   2024-05-13 14:23
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import torch
import torch.nn as nn


class ElmanRNN(nn.Module):
    def __init__(self, input_size, hidden_size, batch_first=False):
        """Initialize the model

        Args:
            input_size (int): the size of the input vectors
            hidden_size (int): the size of the hidden state vectors
            output_dim (int): the size of the output vectors
        """
        super(ElmanRNN, self).__init__()
        self.rnn_cell = nn.RNNCell(input_size=input_size, hidden_size=hidden_size)
        self.batch_first = batch_first
        self.hidden_size = hidden_size

    def _initial_hidden(self, batch_size):
        return torch.zeros((batch_size, self.hidden_size))

    def forward(self, x_in, initial_hidden=None):
        """

        Args:
            x_in (torch.Tensor): an input data tensor. x_in.shape should be (batch, seq_size, feature_size) if batch_first is True, otherwise (seq_size, batch, feature_size).
            initial_hidden (torch.Tensor): the initial hidden state for the RNN. If None, a tensor of zeros will be used.
        Returns:
            hiddens (torch.Tensor): the outputs of the RNN for each step. hiddens.shape should be (batch, seq_size, hidden_size) if batch_first is True, otherwise (seq_size, batch, hidden_size).

        """
        if self.batch_first:
            batch_size, seq_size, feature_size = x_in.size()
            x_in = x_in.permute(1, 0, 2)
        else:
            seq_size, batch_size, feature_size = x_in.size()
        hiddens = []

        if initial_hidden is None:
            initial_hidden = self._initial_hidden(batch_size)
            initial_hidden = initial_hidden.to(x_in.device)
        hidden_t = initial_hidden
        for t in range(seq_size):
            hidden_t = self.rnn_cell(x_in[t], hidden_t)
            hiddens.append(hidden_t)
        hiddens = torch.stack(hiddens)

        if self.batch_first:
            hiddens = hiddens.permute(1, 0, 2)

        return hiddens
