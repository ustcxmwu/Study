#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   batch_gen_multi_thread.py
@Time    :   2024-11-29 15:14
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :
"""

import queue
import threading
import time


def producer(q):
    for i in range(30):  # 生成30条数据
        item = f"data_{i}"
        print(f"Producing {item}")
        q.put(item)  # 将单条数据放入队列
        time.sleep(0.5)  # 模拟生产延迟
    for _ in range(NUM_CONSUMERS):  # 发送结束信号给所有消费者
        q.put(None)


NUM_CONSUMERS = 1  # 假设有3个消费者线程


def data_generator(q, batch_size):
    while True:
        batch = []
        for _ in range(batch_size):
            item = q.get()
            if item is None:  # 生产结束信号
                return
            batch.append(item)
        yield batch  # 生成器返回一批数据，长度为batch_size


def consumer(generator):
    for batch in generator:
        print(f"Consuming {batch}")
        time.sleep(2)  # 模拟处理延迟


def main():
    q = queue.Queue()
    batch_size = 5

    # 创建并启动生产者线程
    producer_thread = threading.Thread(target=producer, args=(q,))
    producer_thread.start()

    # 创建消费者线程
    consumer_threads = []
    for _ in range(NUM_CONSUMERS):
        gen = data_generator(q, batch_size)
        # t = threading.Thread(target=consumer, args=(gen,))
        # consumer_threads.append(t)
        # t.start()
        consumer(gen)

    # 等待生产者线程完成
    producer_thread.join()

    # 等待所有消费者线程完成
    for t in consumer_threads:
        t.join()


if __name__ == "__main__":
    main()
