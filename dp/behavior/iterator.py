from abc import ABC, abstractmethod


class Iterator(ABC):

    @abstractmethod
    def first(self):
        pass

    @abstractmethod
    def next(self):
        pass

    @abstractmethod
    def is_done(self):
        pass

    @abstractmethod
    def curr_item(self):
        pass


class Aggregate(ABC):

    @abstractmethod
    def create_iterator(self):
        pass


class ConcreteIterator(Iterator):

    def __init__(self, aggregate):
        self.aggregate = aggregate
        self.curr = 0

    def first(self):
        return self.aggregate[0]

    def next(self):
        ret = None
        self.curr += 1
        if self.curr < len(self.aggregate):
            ret = self.aggregate[self.curr]
        return ret

    def is_done(self):
        return True if self.curr + 1 >= len(self.aggregate) else False

    def curr_item(self):
        return self.aggregate[self.curr]


class ConcreteAggregate(Aggregate):

    def __init__(self):
        self.ilist = []

    def create_iterator(self):
        return ConcreteIterator(self)


class ConcreteIteratorDesc(Iterator):

    def __init__(self, aggregate):
        self.aggregate = aggregate
        self.curr = len(self.aggregate) - 1

    def first(self):
        return self.aggregate[-1]

    def next(self):
        ret = None
        self.curr -= 1
        if self.curr >= 0:
            ret = self.aggregate[self.curr]
        return ret

    def is_done(self):
        return True if self.curr -1 < 0 else False

    def curr_item(self):
        return self.aggregate[self.curr]


def main():
    ca = ConcreteAggregate()
    ca.ilist.append("小菜")
    ca.ilist.append("大鸟")
    ca.ilist.append("小鸟")
    ca.ilist.append("大菜")
    itor = ConcreteIterator(ca.ilist)
    print(itor.first())
    while not itor.is_done():
        print(itor.next())
    print("逆序".center(80, "="))
    itor_desc = ConcreteIteratorDesc(ca.ilist)
    print(itor_desc.first())
    while not itor_desc.is_done():
        print(itor_desc.next())




if __name__ == "__main__":
    main()