class Singleton1(object):
    def __new__(cls, *args, **kwargs):
        if not hasattr(cls, 'instance'):
            cls.instance = super().__new__(cls)
        return cls.instance


def test_singleton1():
    s = Singleton1()
    s1 = Singleton1()
    assert s == s1


class LazySingleton(object):
    __instance = None

    def __init__(self):
        if not LazySingleton.__instance:
            print('__init__ method called.')
        else:
            print('Instance already created.', self.get_instance())

    @classmethod
    def get_instance(cls):
        if not cls.__instance:
            cls.__instance = LazySingleton()
        return cls.__instance


def test_lazy_singleton():
    s = LazySingleton()
    # assert s.__instance == None
    LazySingleton.get_instance()
    s1 = LazySingleton()
    assert s1.get_instance() == s.get_instance()


class MetaSingleton(type):
    __instances = {}

    def __call__(cls, *args, **kwargs):
        if cls not in cls.__instances:
            cls.__instances[cls] = super(MetaSingleton, cls).__call__(*args, **kwargs)
        return cls.__instances[cls]


class Logger(metaclass=MetaSingleton):
    pass


def test_logger():
    log1 = Logger()
    log2 = Logger()
    assert log1 == log2
