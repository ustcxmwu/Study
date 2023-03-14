class ListMetaClass(type):

    def __new__(cls, name, bases, attrs):
        attrs['add'] = lambda self, value: self.append(value)
        return type.__new__(cls, name, bases, attrs)


class MyList(list, metaclass=ListMetaClass):
    pass


if __name__ == '__main__':
    l = MyList()
    l.add(1)
    l.add(2)
    print(l)
