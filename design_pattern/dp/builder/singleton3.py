class SingletonMeta(type):
    def __init__(cls, *args, **kwargs):
        cls.__instance = None
        super(SingletonMeta, cls).__init__(*args, **kwargs)

    def __call__(cls, *args, **kwargs):
        if cls.__instance is None:
            cls.__instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls.__instance


class Singleton(metaclass=SingletonMeta):
    pass


def main():
    t1 = Singleton()
    t2 = Singleton()


if __name__ == "__main__":
    main()
