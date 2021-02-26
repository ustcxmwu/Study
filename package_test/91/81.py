#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

def foo():
    sum = 0
    for i in range(100):
        sum += i
    return sum

if __name__ == '__main__':
    import cProfile
    cProfile.run("foo()")