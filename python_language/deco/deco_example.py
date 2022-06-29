# -*- coding: utf-8 -*-
# @Time    : 2022/6/29 9:56
# @Author  : Xiaomin Wu
# @Email   : ustcxmwu@gmai.com
# @Desc    :



def deco(func):
    def inner():
        print("running inner")
        func()
    return inner



@deco
def target():
    print("running target")


if __name__ == '__main__':
    target()