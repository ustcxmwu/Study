#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.


def consumer():
    while True:
        line = yield
        print(line.upper())


def producer():
    with open("57.py", mode='r') as f:
        for i, line in enumerate(f):
            yield line
            print("processed line:{}".format(line))


if __name__ == '__main__':
    c = consumer()
    next(c)
    for line in producer():
        c.send(line)

    import gevent
    from gevent import socket
    urls = ['www.google.com', 'www.example.com', 'www.python.org']
    jobs = [gevent.spawn(socket.gethostbyname, url) for url in urls]
    gevent.joinall(jobs, timeout=2)
    for job in jobs:
        print(job.value)

