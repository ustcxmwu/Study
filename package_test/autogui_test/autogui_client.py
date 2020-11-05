#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import pyautogui


if __name__ == '__main__':
    w, h = pyautogui.size()
    print(w, h)
    print(pyautogui.position())
    pyautogui.locateOnScreen("sign.png")
    for i in range(4):
        pyautogui.moveRel(100, 0, duration=0.25)
        pyautogui.moveRel(0, 100, duration=0.25)
        pyautogui.moveRel(-100, 0, duration=0.25)
        pyautogui.moveRel(0, -100, duration=0.25)

