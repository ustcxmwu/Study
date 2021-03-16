from abc import ABCMeta, abstractmethod


class Publisher(object):

    def __init__(self):
        self.observers = []

    def add(self, observer):
        if observer not in self.observers:
            self.observers.append(observer)
        else:
            print("Failed to add:{}".format(observer))

    def remove(self, observer):
        try:
            self.observers.remove(observer)
        except ValueError:
            print("Failed to remove: {}".format(observer))

    def notify(self):
        [observer.notify(self) for observer in self.observers]


class DefaultFormatter(Publisher):

    def __init__(self, name):
        super().__init__()
        self.name = name
        self._data = 0

    def __str__(self):
        return "{}: {} has data = {}".format(type(self).__name__, self.name, self._data)

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        try:
            self._data = int(value)
        except ValueError as e:
            print("Error {}".format(e))
        else:
            self.notify()


class Oberver(metaclass=ABCMeta):
    @abstractmethod
    def notify(self, publisher: Publisher):
        pass


class HexFormatter(Oberver):

    def notify(self, publisher):
        print("{}: {} has now hex data = {}".format(type(self).__name__, publisher.name, hex(publisher.data)))


class BinaryFormatter(Oberver):

    def notify(self, publisher):
        print("{}: {} has now bin data = {}".format(type(self).__name__, publisher.name, bin(publisher.data)))


def main():
    df = DefaultFormatter("test1")
    print(df)

    print()
    hf = HexFormatter()
    df.add(hf)
    df.data = 3
    print(df)

    print()
    bf = BinaryFormatter()
    df.add(bf)
    df.data = 21
    print(df)

    print()
    df.remove(hf)
    df.data = 40
    print(df)

    print()
    df.remove(hf)
    df.add(bf)

    df.data = "hello"
    print(df)

    print()
    df.data = 15.8
    print(df)


if __name__ == "__main__":
    main()
