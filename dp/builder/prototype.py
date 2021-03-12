#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

import copy
from collections import OrderedDict


class Book(object):

    def __init__(self, name, authors, price, **rest):
        self.name = name
        self.authors = authors
        self.price = price
        self.__dict__.update(rest)

    def __str__(self):
        mylist = []
        ordered = OrderedDict(sorted(self.__dict__.items()))
        for i in ordered.keys():
            mylist.append("{}: {}".format(i, ordered[i]))
            if i == "price":
                mylist.append("$")
            mylist.append("\n")
        return "".join(mylist)


class Prototype(object):
    def __init__(self):
        self.objects = dict()

    def register(self, identifier, obj):
        self.objects[identifier] = obj

    def unregister(self, identifier):
        del self.objects[identifier]

    def clone(self, identifier, **attr):
        found = self.objects.get(identifier)
        if not found:
            raise ValueError("Incorrect object identifier: {}".format(identifier))
        obj = copy.deepcopy(found)
        obj.__dict__.update(attr)
        return obj


def main():
    b1 = Book("C Language", ("Zhao", "Qian", "Sun"), price=100, publisher="Prentice Hall", length=228)
    prototype = Prototype()
    cid = "k&r first"
    prototype.register(cid, b1)
    b2 = prototype.clone(cid, name="The C Language", price=50, length=255)
    for i in (b1, b2):
        print(i)
    print("ID b1: {} != ID b2 {}".format(id(b1), id(b2)))


if __name__ == "__main__":
    main()


