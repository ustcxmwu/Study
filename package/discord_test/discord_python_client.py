#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   discord_python_client.py
@Time    :   2023-11-16 14:04
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2023, Wu Xiaomin
@Desc    :   
"""

from discord import Application, User, Guild, Channel, Stage, Webhook


if __name__ == "__main__":
    token = "MTEwMzkyNzI5MTE0ODUwOTIwNA.G0WuOq.FHMRQQVWEghSPfkh1fJcnwiiWNeVg0PLvWjg8w"
    channel = Channel(token=token)
    print(channel)
