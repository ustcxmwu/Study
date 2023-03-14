#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import logbook
from logbook.queues import ZeroMQSubscriber
from logbook import Logger, StderrHandler, FileHandler
import sys
import time

if __name__ == '__main__':
    addr = 'tcp://127.0.0.1:5053'
    print("ZeroMQSubscriber begin with address {}".format(addr))
    subscriber = ZeroMQSubscriber(addr, multi=True)

    target_handlers = logbook.NestedSetup([
        StderrHandler(level=logbook.INFO,
                      format_string='{record.time:%Y-%m-%d %H:%M:%S}|{record.level_name}|{record.message}'),
        FileHandler('test_logbook_mp.log', mode='w', level=logbook.DEBUG, bubble=True)
    ])
    with target_handlers:
        subscriber.dispatch_forever()

    # try:
    #     i=0
    #     while True:
    #         i += 1
    #         record = subscriber.recv(2)
    #         if not record:
    #             pass # timeout
    #         else:
    #             print("got message!")
    #             log.handle(record)
    # except KeyboardInterrupt:
    #     print("C-C caught, program end after {} iterations".format(i))
    # handler.pop_application()
