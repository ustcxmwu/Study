#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   transformer_client.py
@Time    :   2024-05-13 16:19
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import torch


def main():
    model = torch.nn.Transformer()
    print(model.encoder.layers[0])


if __name__ == '__main__':
    main()
