

class MyType(type):

    def __init__(self, a, b, c):
        print("meta class")
        print(a)
        print(b)
        print(c)

    def __call__(self, *args, **kwargs):
        obj = object.__new__(self)
        self.__init__(obj, *args, **kwargs)
        return obj


class Foo(metaclass=MyType):
    def __init__(self, name):
        self.name = name


if __name__ == '__main__':
    f1 = Foo("Alex")
    print(f1)
    print(f1.__dict__)