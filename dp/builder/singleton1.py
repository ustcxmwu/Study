

class Singleton(object):
    instance = None
    def __new__(cls, *args, **kwargs):
        if cls.instance is None:
            cls.instance = super().__new__(cls, *args, **kwargs)
        return cls.instance


def main():
    t1 = Singleton()
    t2 = Singleton()
    assert id(t1) == id(t2)


if __name__ == "__main__":
    main()