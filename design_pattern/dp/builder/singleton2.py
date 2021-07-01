import threading


class Singleton(object):
    vars = {}
    single_lock = threading.Lock()
    def __new__(cls, *args, **kwargs):
        if cls in cls.vars:
            return cls.vars[cls]
        cls.single_lock.acquire()
        try:
            if cls in cls.vars:
                return cls.vars[cls]
            cls.vars[cls] = super().__new__(cls, *args, **kwargs)
            return cls.vars[cls]
        finally:
            cls.single_lock.release()


def main():
    t1 = Singleton()
    t2 = Singleton()
    assert id(t1) == id(t2)


if __name__ == "__main__":
    main()
