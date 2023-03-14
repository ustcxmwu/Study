# -*- encoding: utf-8 -*-
"""
@File    : client.py
@Time    : 2020-07-03 16:31
@Author  : wuxiaomin
@Email   : xmwu@mail.ustc.edu.cn
@Description:
"""

import pickle

import zerorpc

from server import Entity

if __name__ == '__main__':
    c = zerorpc.Client()
    c.connect("tcp://127.0.0.1:4242")
    print(c.hello("RPC"))
    e = Entity("xiaomin", 30)
    print(c.hello_entity(pickle.dumps(e)))
