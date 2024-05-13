from easydict import EasyDict as edict

if __name__ == '__main__':
    a = edict({"a": 1, "b": 2, "c": {"d": 4, "e": 5}})
    print(a.a)
    print(a.c.d)

    print(a)
