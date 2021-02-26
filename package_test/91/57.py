#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import math


def len(point):
    return math.sqrt(point.x * point.x + point.y * point.y)


class RTriangle(object):
    def __init__(self, x, y):
        self.x = x
        self.y = y

RTriangle.len = len
rt = RTriangle(3, 4)
print(rt.len())


value = 'default global'

class Test(object):
    def __init__(self, part):
        value = part + "====="
        self.value = value

    def show(self):
        print(self.value)
        print(value)

a = Test("instance")
a.show()

if __name__ == '__main__':
    a = Test("instance")
    a.show()
