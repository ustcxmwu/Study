

class Singleton1(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super(Singleton1, cls).__new__(cls)
        return cls.instance


class Singleton2(object):
    __instance = None
    def __init__(self):
        if not Singleton2.__instance:
            print('__init__ method called.')
        else:
            print('Instance already created.', self.getInstance())

    @classmethod
    def getInstance(cls):
        if not cls.__instance:
            cls.__instance = Singleton2()
        return cls.__instance






if __name__ == '__main__':
    # s = Singleton1()
    # print("Singleton1 created ", s)
    #
    # s1 = Singleton1()
    # print("Singleton1 created ", s1)

    s3 = Singleton2()
    print('Instance created.', s3.getInstance())
    s4 = Singleton2()
