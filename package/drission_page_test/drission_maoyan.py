#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   drission_maoyan.py
@Time    :   2024-04-15 14:56
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
from DrissionPage import ChromiumPage
from DataRecorder import Recorder


def maoyan():
    global page, num, score, title, star, time
    # 创建页面对象
    page = ChromiumPage()
    # 创建记录器对象
    recorder = Recorder('data.csv')
    # 访问网页
    page.get('https://www.maoyan.com/board/4')
    while True:
        # 遍历页面上所有 dd 元素
        for mov in page.eles('t:dd'):
            # 获取需要的信息
            num = mov('t:i').text
            score = mov('.score').text
            title = mov('@data-act=boarditem-click').attr('title')
            star = mov('.star').text
            time = mov('.releasetime').text
            # 写入到记录器
            recorder.add_data((num, title, star, time, score))

        # 获取下一页按钮，有就点击
        btn = page('下一页', timeout=2)
        if btn:
            btn.click()
            page.wait.load_start()
        # 没有则退出程序
        else:
            break
    recorder.record()


if __name__ == '__main__':
    maoyan()