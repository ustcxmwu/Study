def singleton(cls, *args, **kwargs):
    instance = {}

    def get_instance():
        if cls not in instance:
            instance[cls] = cls(*args, **kwargs)
        return instance[cls]

    return get_instance


@singleton
class A(object):
    pass


def main():
    t1 = A()
    t2 = A()
    print(t1)
    print(t2)
    assert id(t1) == id(t2)


if __name__ == "__main__":
    main()
