#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import time
import argparse
import multiprocessing as mp
import os
import logbook
from logbook.queues import MultiProcessingHandler, MultiProcessingSubscriber


def waitForTime(name, secs, pid):
    for tick in reversed(range(0, secs)):
        logbook.debug('-- {} {} waiting at {}', name, pid, tick)
        time.sleep(1)


def setup_logging(queue, level=logbook.DEBUG):
    handler = MultiProcessingHandler(queue)
    handler.push_application()


def foo(*argv):
    setup_logging(argv[2])
    try:
        name = str(argv[0])
        secs = argv[1]
        my_pid = os.getpid()
        logbook.info('STARTED {} with pid {}', name, my_pid)
        waitForTime(name, secs, my_pid)
    except:
        logbook.critical('Exception killed {} with pid {}', name, my_pid)
        logbook.info('EXITING {} with pid {}', name, my_pid)


def main():
    # mp.set_start_method('spawn')
    parser = argparse.ArgumentParser(description="Run a log demo")
    parser.add_argument("--logLevel", nargs='?', const=1, type=str, default="DEBUG",
                        help="logging level {DEBUG|INFO|WARNING|ERROR|CRITICAL}")
    args = parser.parse_args()

    queue = mp.Queue()

    setup_logging(queue)
    logbook.info("main starting: Args=%s" % args)

    plist = []
    plist.append(mp.Process(target=foo, args=('moe', 10, queue)))
    plist.append(mp.Process(target=foo, args=('larry', 6, queue)))
    plist.append(mp.Process(target=foo, args=('curly', 12, queue)))
    plist.append(mp.Process(target=foo, args=('shemp', 4, queue)))

    if 0 < len(plist):
        target_handlers = logbook.NestedSetup([
            logbook.NullHandler(),
            logbook.StderrHandler(level=logbook.INFO,
                format_string='{record.time:%Y-%m-%d %H:%M:%S}|{record.level_name}|{record.message}'),
            logbook.FileHandler('test_logbook_mp.log', mode='w', level=logbook.DEBUG, bubble=True)
        ])

        sub = MultiProcessingSubscriber(queue)
        sub.dispatch_in_background(target_handlers)

        logbook.info("Will start %d processes" % len(plist))
        for p in plist:
            p.start()
        logbook.debug('main waiting for all processes to stop')
        for p in plist:
            p.join()
    else:
        logbook.warn('main has no processes to run')
        logbook.info("main exiting")


if __name__ == '__main__':
    # todo: we want a new log file every run with a filename format like
    # 'log'+'.'+UTC YYYY-MM-DD:HH:MM+'
    main()


