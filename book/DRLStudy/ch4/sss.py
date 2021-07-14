
class A(object):
    def __init__(self, a, b):
        self.a = a
        self.b = b

    @classmethod
    def make_A(cls, a, b):
        t = A(a, b)
        return t


if __name__ == '__main__':
    s = A.make_A(1, 2)
    print(s.a, s.b)
