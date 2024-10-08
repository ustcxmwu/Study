#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   user_path_client.py
@Time    :   2024-07-15 10:43
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
from userpath import UserPathTracker, MemoryBackend

if __name__ == '__main__':
    # 初始化UserPathTracker和存储后端
    tracker = UserPathTracker(backend=MemoryBackend())

    # 模拟用户行为路径
    tracker.log_event("login", {"user_id": "123"})
    tracker.log_event("view_page", {"page": "home"})
    tracker.log_event("click_button", {"button": "profile"})
    tracker.log_event("view_page", {"page": "profile"})
    tracker.log_event("logout", {"user_id": "123"})

    # 生成路径可视化
    tracker.visualize_path()

    # 输出路径统计信息
    print(tracker.get_path_statistics())
