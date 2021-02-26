#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

class Singleton(type):
    def __init__(self, *args, **kwargs):
        print("__init__")
        self.__instance = None
        super(Singleton, self).__init__(*args, **kwargs)

    def __call__(self, *args, **kwargs):
        print("__call__")
        if self.__instance is None:
            self.__instance = super(Singleton, self).__call__(*args, **kwargs)
        return self.__instance


class Foo(metaclass=Singleton):
    pass



class Meta(type):
    def f1(cls):
        print("this is f1")

class SubMeta(Meta):
    def f2(cls):
        print("this is f2")

class Test(metaclass=SubMeta):
    @classmethod
    def f3(cls):
        print("this is f3")

t = Test()

Meta.f1(Test)
SubMeta.f1(Test)
SubMeta.f2(Test)
# Test.f2()
# t.f2()
# Test.f3()
# t.f3()


class M1(type):
    def __new__(meta, name , bases, attrs):
        return super(M1, meta).__new__(meta, name, bases, attrs)

class M2(type):
    def __new__(meta, name, bases, attrs):
        return super(M2, meta).__new__(meta, name, bases, attrs)

class C1(metaclass=M1):
    pass

class C2(metaclass=M2):
    pass

class Sub(C1, C2):
    pass




if __name__ == '__main__':
    a = Foo()
    b = Foo()
    print(id(a))
    print(id(b))

    t = Test()

    Meta.f1(Test)
    SubMeta.f1(Test)
    SubMeta.f2(Test)
    Test.f2()
    # t.f2()
    Test.f3()
    t.f3()

    s = Sub()
