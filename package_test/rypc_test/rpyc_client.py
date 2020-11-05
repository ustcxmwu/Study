#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import rpyc
import time
from pathlib import Path


if __name__ == '__main__':
    # conn = rpyc.connect("apps-rl.danlu.netease.com", 39217)
    # conn = rpyc.connect("localhost", 9000)
    # curr = time.time()
    # conn.root.set_time(curr)
    # print(curr)
    # time.sleep(2)
    # t, i = conn.root.get_interval()
    # print(t, i)
    # conn.close()

    # conn = rpyc.connect("apps-rl.danlu.netease.com", 39217)
    # with open("rpyc_client.py", "r") as f:
    #     content = f.readlines()
    #
    # conn.root.save_lines(Path("./root/rpyc/client.py"), content)
    # conn.close()

    team_cnts = {0: 1, 1: 1}
    print(any(team_cnts.values() == 1))

