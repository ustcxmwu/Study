# coding=utf-8

__author__ = 'Administrator'

__doc__ = '''
pythonwin中win32gui的用法
本文件演如何使用win32gui来遍历系统中所有的顶层窗口，
并遍历所有顶层窗口中的子窗口
'''

import win32gui
import win32com.client
import time
import win32con
import pythoncom

from pprint import pprint


def show_window_attr(hWnd):
    '''
    显示窗口的属性
    :return:
    '''
    if not hWnd:
        return

    # 中文系统默认title是gb2312的编码
    title = win32gui.GetWindowText(hWnd)
    # title = gbk2utf8(title)
    clsname = win32gui.GetClassName(hWnd)

    print('窗口句柄:%s ' % (hWnd))
    print('窗口标题:%s' % (title))
    print('窗口类名:%s' % (clsname))


def show_windows(hWndList):
    for h in hWndList:
        show_window_attr(h)


def demo_top_windows():
    '''
    演示如何列出所有的顶级窗口
    :return:
    '''
    hWndList = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), hWndList)
    show_windows(hWndList)

    return hWndList


def demo_child_windows(parent):
    '''
    演示如何列出所有的子窗口
    :return:
    '''
    if not parent:
        return

    hWndChildList = []
    win32gui.EnumChildWindows(parent, lambda hWnd, param: param.append(hWnd), hWndChildList)
    show_windows(hWndChildList)
    return hWndChildList


def get_wyx_course_windows():
    hWndList = []
    win32gui.EnumWindows(lambda hWnd, param: param.append(hWnd), hWndList)

    targetWnds = []
    for h in hWndList:
        if not h:
            continue

        title = win32gui.GetWindowText(h)
        # title = gbk2utf8(title)
        clsname = win32gui.GetClassName(h)

        if title.find("Google Chrome") >= 0 and title.find("云课堂") >= 0:
            targetWnds.append(h)

    # show_windows(targetWnds)
    return targetWnds


def activate_window(hWnd):
    # win32gui.BringWindowToTop(hWnd)
    win32gui.SendMessage(hWnd, win32con.WM_SYSCOMMAND, win32con.SC_RESTORE, 0)
    # Send an alt event first, otherwise an error will be reported and
    # the subsequent settings will be invalid: pywintypes.error:
    # (0,'SetForegroundWindow','No error message is available')
    shell = win32com.client.Dispatch("WScript.Shell")
    shell.SendKeys('%')
    # Set as the current active window
    win32gui.SetForegroundWindow(hWnd)


def loop_all_windows(hWndList):
    for hWnd in hWndList:
        if hWnd:
            print("---共{}个窗口---".format(len(hWndList)))
            print("{}: 激活以下窗口".format(time.asctime(time.localtime(time.time()))))
            show_window_attr(hWnd)
            activate_window(hWnd)
            time.sleep(5)


if __name__ == "__main__":
    while True:
        try:
            targetWnds = get_wyx_course_windows()
            loop_all_windows(targetWnds)
        except Exception as e:
            print(e)

