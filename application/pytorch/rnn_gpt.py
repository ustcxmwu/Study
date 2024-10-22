#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   rnn_gpt.py
@Time    :   2024-10-17 14:41
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""

import torch
import torch.nn as nn
import torch.optim as optim

input_size = 5
hidden_size = 10
output_size = 1
num_layers = 1
seq_len = 6


class SimpleRNN(nn.Module):

    def __init__(self, input_size, hidden_size, output_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        h_0 = torch.zeros(num_layers, x.size(0), hidden_size)

        out, _ = self.rnn(x, h_0)

        out = self.fc(out[:, -1, :])
        return out


if __name__ == '__main__':
    model = SimpleRNN(input_size, hidden_size, output_size, num_layers)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    batch_size = 3
    x_train = torch.randn(batch_size, seq_len, input_size)
    y_train = torch.randn(batch_size, output_size)

    num_epochs = 100
    for epoch in range(num_epochs):
        outputs = model(x_train)
        loss = criterion(outputs, y_train)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")
    print(f"final loss: {loss.item():.4f}")
