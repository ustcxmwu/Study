#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   tqdm_client.py
@Time    :   2026-01-21 10:46
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""

from tqdm import tqdm
import enlighten
import time

if __name__ == '__main__':
    with tqdm(total=100, desc="任务 A", position=0) as p1, \
            tqdm(total=100, desc="任务 B", position=1) as p2:
        for i in range(100):
            print( "xxxxxxxxxxxxx" )
            time.sleep(0.05)
            p1.update(1)
            p2.update(1)

    # # 1. 初始化管理器
    # manager = enlighten.get_manager()
    #
    # # 2. 创建多个计数器（进度条）
    # # color 参数可以直接设置颜色（如 'green', 'red', 'bold_blue'）
    # pbar1 = manager.counter(total=100, desc="下载任务", unit="MB", color="green")
    # pbar2 = manager.counter(total=100, desc="解压任务", unit="files", color="red")
    #
    # for i in range(100):
    #     time.sleep(0.05)
    #     pbar1.update()
    #     print("xxxxxxxxxxxxx")
    #     # 模拟任务 B 进度不同步
    #     if i % 2 == 0:
    #         pbar2.update()
    #     print("xxxxxxxxxxxxx")
    #
    # # 3. 结束后停止管理器（清理屏幕空间）
    # manager.stop()