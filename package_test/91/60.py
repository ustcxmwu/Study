#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

class A(object):
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        print("call __getattr__: {}".format(name))

a = A("attribute")
print(a.name)
print(a.test)

class B(object):
    def __init__(self, name):
        self.name = name

    def __getattr__(self, name):
        print("call __getattr__: {}".format(name))

    def __getattribute__(self, name):
        try:
            return self.__dict__[name]
        except KeyError:
            return "default"

# b = B("attribute")
# print(b.name)
# print(b.test)



class C(object):
    def __init__(self):
        self.x = None

    @property
    def a(self):
        print("using property to access attribute")
        if self.x is None:
            print("return value")
            return "a"
        else:
            print("error occured")
            raise AttributeError

    @a.setter
    def a(self, value):
        self.x = value

    def __getattr__(self, name):
        print("using __getattr__ to access attribute")
        print("attribute name: {}".format(name))
        return "b"

    def __getattribute__(self, name):
        print("using __getattribute__ to access attribute")
        return object.__getattribute__(self, name)

c = C()
print(c.a)
print("-----")
c.a = 1
print(c.a)
print("------")



if __name__ == '__main__':
    pass
