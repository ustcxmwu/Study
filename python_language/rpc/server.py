# -*- encoding: utf-8 -*-
"""
@File    : server.py
@Time    : 2020-07-03 16:30
@Author  : wuxiaomin
@Email   : xmwu@mail.ustc.edu.cn
@Description:
"""

import zerorpc
import msgpack
import pickle


class HelloRpc(object):

    def hello(self, name):
        print(name)
        return "Hello"

    def hello_entity(self, e):
        entity = pickle.loads(e)

        print(entity.name)
        print(entity.age)
        return "{},{}".format(entity.name, entity.age)


class Entity(object):

    def __init__(self, name, age):
        self.name = name
        self.age = age




if __name__ == '__main__':
    s = zerorpc.Server(HelloRpc())
    s.bind("tcp://127.0.0.1:4242")
    s.run()

