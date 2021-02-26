#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import gc
import pprint

class Leak(object):
    def __init__(self):
        print("object with id {} was born".format(id(self)))

if __name__ == '__main__':
    collected = gc.collect()
    print("Garbage collector before running: collected {} objects.".format(collected))
    A = Leak()
    B = Leak()
    A.b = B
    B.a = A
    A = None
    B = None
    collected = gc.collect()
    print("Garbage collector after running: collected {} objects.".format(collected))

