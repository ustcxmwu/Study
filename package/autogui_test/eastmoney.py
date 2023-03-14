#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   eastmoney.py
@Time    :   2022-10-17 18:21
@Author  :   Wu Xiaomin <wuxiaomin@pandadastudio.com>
@Version :   1.0
@License :   (C)Copyright 2022, Wu Xiaomin
@Desc    :   
"""

if __name__ == '__main__':
    import pyautogui as gui, time

    screenWidth, screenHeight = gui.size()
    gui.moveTo(0, screenHeight)
    gui.click()

    gui.typewrite('Firefox', interval=0.25)
    gui.press('enter')