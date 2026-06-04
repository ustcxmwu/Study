#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   enlighten_client1.py
@Time    :   2026-05-26 15:27
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2026, Wu Xiaomin
@Desc    :   
"""

import os
import time

from enlighten import Manager, get_manager

os.environ["TERM"] = "xterm-256color"


def fun1():
    mgr = Manager()

    # 创建两个进度条
    outer = mgr.counter(total=5, desc="外层循环", unit="轮")
    inner = mgr.counter(total=3, desc="内层循环", unit="次")

    for i in range(5):
        for j in range(3):
            time.sleep(0.2)
            inner.update()
        outer.update()
        inner.reset()  # 内层重置，下次继续用


def fun2():
    with get_manager() as manager:
        # 创建两个不同的进度条
        task1 = manager.counter(total=10, desc="大任务 A", unit="steps", color="red")
        task2 = manager.counter(total=50, desc="子任务 B", unit="files", color="green")

        for i in range(10):
            # 即使在循环中直接使用 print，进度条也不会错位
            # print(f"正在启动第 {i+1} 阶段的外部常规日志...")

            task1.update()

            # 重置并运行子任务
            task2.count = 0
            task2.total = 50
            for _ in range(50):
                time.sleep(0.01)
            task2.update()


def main():
    # 1. 统一由一个 manager 管理
    import sys
    with get_manager(stream=sys.stdout) as manager:

        # 2. 创建外层进度条 (位于上方)
        outer_bar = manager.counter(total=3, desc="Outer Loop", unit="epochs", color="red")

        # 3. 创建内层进度条 (位于下方)
        inner_bar = manager.counter(total=100, desc="Inner Loop", unit="steps", color="green")

        for epoch in range(3):
            # 每次外层循环开始时，重置内层进度条的计数
            inner_bar.count = 0
            for step in range(100):
                time.sleep(0.01)
                # 更新内层
                inner_bar.update()

            # 更新外层
            outer_bar.update()


def fun3():
    # 1. 初始化管理器
    with get_manager() as manager:

        # 2. 创建主进度条（总任务数），设置颜色为红色
        total_tasks = 4
        main_pbar = manager.counter(
            total=total_tasks,
            desc="[Total Progress]",
            unit="tasks",
            color="red"
        )

        # 3. 模拟多个子任务依次执行
        for task_idx in range(1, total_tasks + 1):

            # 打印日志（enlighten 会自动将其推到进度条上方，不会弄乱终端）
            print(f"[LOG] 正在启动任务 #{task_idx}...")

            # 4. 动态创建子进度条
            # leave=False 表示该子任务完成后，其进度条会从终端隐去，保持界面干净
            sub_steps = 10
            sub_pbar = manager.counter(
                total=sub_steps,
                desc=f" └─ Task #{task_idx}",
                unit="steps",
                color="cyan",
                leave=False
            )

            # 5. 执行子任务逻辑
            for _ in range(sub_steps):
                time.sleep(0.1)  # 模拟耗时操作
                sub_pbar.update()  # 更新子进度

            # 6. 子任务完成后关闭它（触发 leave=False 的隐去效果）
            sub_pbar.close()

            # 7. 更新主进度条
            main_pbar.update()
            print(f"[LOG] 任务 #{task_idx} 已完成！")


if __name__ == "__main__":
    fun3()
