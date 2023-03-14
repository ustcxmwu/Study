#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import rpyc


def on_file_changed(old_stat, new_stat):
    print("file changes")
    print("old stat: {}".format(old_stat))
    print("new stat: {}".format(new_stat))


if __name__ == '__main__':
    conn = rpyc.connect("localhost", 9001)
    monn = conn.root.FileMonitor("F:/Dir/a.txt", on_file_changed)
