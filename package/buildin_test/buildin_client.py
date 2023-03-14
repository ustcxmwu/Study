#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import heapq
from collections import ChainMap, namedtuple
from threading import Thread


def heapq_test():
    a = [4, 3, 2, 1]
    print(a)
    heapq.heapify(a)
    print(a)
    heapq.heappush(a, 5)
    print(a)
    heapq.heappush(a, 1)
    print(a)


if __name__ == '__main__':
    heapq_test()

