#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import time
from rpyc import Service
from rpyc.utils.server import ThreadedServer


class TimeService(Service):

    def __init__(self):
        super().__init__()
        self.curr = time.time()

    def exposed_set_time(self, curr):
        self.curr = curr

    def exposed_get_time(self):
        return time.ctime()

    def exposed_get_interval(self):
        print(self.curr)
        now = time.time()
        return now, now - self.curr


if __name__ == '__main__':
    s = ThreadedServer(service=TimeService(), port=9000, auto_register=False)
    s.start()
