#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import keyboard
import yaml
from threading import Thread
import time


def func():
    while True:
        print("XXXX")
        time.sleep(1)




if __name__ == '__main__':
    # recorded = keyboard.record(until='esc')
    # for r in recorded:
    #     print(r)

    # press a to print rk
    # keyboard.add_hotkey('a', lambda: keyboard.write('Geek'))
    # keyboard.add_hotkey('ctrl + shift + a', print, args=('you entered', 'a'))
    # keyboard.add_hotkey('ctrl + shift + b', print, args=('you entered', 'b'))
    # keyboard.add_hotkey('ctrl + shift + c', print, args=('you entered', 'c'))
    # keyboard.add_hotkey('ctrl + shift + c', print, args=('you entered', 'd'))
    t = Thread(target=func, args=())
    t.setDaemon(False)
    t.start()
    with open("keyboard.yml", mode='r') as f:
        config = yaml.safe_load(f)
    for key, val in config.items():
        keyboard.add_hotkey(key, print, args=('you entered', val))

    keyboard.wait('q')