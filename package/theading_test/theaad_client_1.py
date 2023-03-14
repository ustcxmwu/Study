#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import signal, sys


def alarm_handler(*args):
    raise Exception("timeout")

def function_xyz(prompt, timeout):
    signal.signal(signal.SIGALRM, alarm_handler)
    signal.alarm(timeout)
    sys.stdout.write(prompt)
    sys.stdout.flush()
    try:
        text = sys.stdin.readline()
    except:
        text = ""
    signal.alarm(0)
    return text


if __name__ == '__main__':
    function_xyz()