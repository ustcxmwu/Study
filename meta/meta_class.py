
class ListMetaClass(type):
    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)


class MyList(list, metaclass=ListMetaClass):
    pass


if __name__ == '__main__':
    L = MyList()
    L.add(1)
    print(L)
    L.add(2)
    print(L)