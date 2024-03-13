#!/usr/bin/env python
# -*-coding:utf-8 -*-
"""
@Project :   Study
@File    :   weichat.py
@Time    :   2024-02-29 16:37
@Author  :   Wu Xiaomin <>
@Version :   1.0
@License :   (C)Copyright 2024, Wu Xiaomin
@Desc    :   
"""
import time

import pyautogui
import pyperclip


def send_msg(recv, messages):
    pyautogui.hotkey("ctrl", "command", "w")  # ctrl + alt + w 打开微信
    pyautogui.hotkey("command", "f")  # 打开查找
    pyperclip.copy(recv)  # 复制好友昵称到粘贴板
    pyautogui.hotkey("command", "v")  # 模拟键盘 ctrl + v 粘贴
    time.sleep(1)
    pyautogui.press("enter")  # 回车进入好友消息界面
    if not isinstance(messages, list):
        messages = [messages]
    for msg in messages:
        pyperclip.copy(msg)  # 复制需要发送的内容到粘贴板
    pyautogui.hotkey("command", "v")  # 模拟键盘 ctrl + v 粘贴内容
    pyautogui.press("enter")  # 发送消息
    time.sleep(1)  # 每条消息间隔 1 秒


if __name__ == "__main__":
    # send_msg("逍遥游", "hello")
    # handle = pyautogui.getWindowsWithTitle(pyautogui.getActiveWindowTitle())[0]._hWnd
    # print(handle)
    from AppKit import NSWorkspace, NSRunningApplication

    # ws = NSWorkspace.sharedWorkspace()
    #     # runningApps = ws.runningApplications()
    #     # for i in runningApps:
    #     #     print(i.bundleIdentifier())
    #     #     print(i.localizedName())

    wx = NSRunningApplication.runningApplicationsWithBundleIdentifier_(
        "com.tencent.xinWeChat"
    )
    for x in wx:
        print(x.bundleIdentifier())
        x.hide()
