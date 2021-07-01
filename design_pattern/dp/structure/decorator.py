from abc import ABCMeta, abstractmethod


class Beverage(object):
    name = ""
    price = 0.0
    type = "BEVERAGE"

    def get_price(self):
        return self.price

    def set_price(self, price):
        self.price = price

    def get_name(self):
        return self.name


class Coke(Beverage):
    def __init__(self):
        self.name = "coke"
        self.price = 4.0


class Milk(Beverage):

    def __init__(self):
        self.name = "milk"
        self.price = 5.0


class DrinkDecorator(metaclass=ABCMeta):

    @abstractmethod
    def get_name(self):
        pass

    @abstractmethod
    def get_price(self):
        pass


class IceDecorator(DrinkDecorator):

    def __init__(self, beverage):
        self.beverage = beverage

    def get_name(self):
        return self.beverage.get_name() + " + ice"

    def get_price(self):
        return  self.beverage.get_price() + 0.3


class SugarDecorator(DrinkDecorator):

    def __init__(self, beverage):
        self.beverage = beverage

    def get_name(self):
        return self.beverage.get_name() + " + sugar"

    def get_price(self):
        return self.beverage.get_price() + 0.5


def main():
    coke_cola = Coke()
    print("Name:{}".format(coke_cola.get_name()))
    print("Price:{}".format(coke_cola.get_price()))

    ice_coke = IceDecorator(coke_cola)
    print("Name:{}".format(ice_coke.get_name()))
    print("Price:{}".format(ice_coke.get_price()))


if __name__ == "__main__":
    main()



