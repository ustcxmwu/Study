#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
from logbook.queues import ZeroMQHandler
from logbook import Logger
import time


if __name__ == '__main__':
    addr='tcp://127.0.0.1:5053'
    handler = ZeroMQHandler(addr)
    time.sleep(0.25)

    log = Logger("myLogbook")
    handler.push_application()
    log.info("start of program")
    log.info("end of program")
    handler.close()
    handler.pop_application()


