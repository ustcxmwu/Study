#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
from pyparsing import Word, OneOrMore, Optional, Group, Suppress, alphanums


class Gate(object):

    def __init__(self):
        self.is_open = False

    def __str__(self):
        return "open" if self.is_open else "closed"

    def open(self):
        print("opening the gate")
        self.is_open = True

    def close(self):
        print("closing the gate")
        self.is_open = False


class Garage(object):

    def __init__(self):
        self.is_open = False

    def __str__(self):
        return "open" if self.is_open else "closed"

    def open(self):
        print("opening the garage")
        self.is_open = True

    def close(self):
        print("closing the garage")
        self.is_open = False


class Aircondition(object):

    def __init__(self):
        self.is_on = False

    def __str__(self):
        return "on" if self.is_on else "off"

    def turn_on(self):
        print("turning on the aircondition")
        self.is_on = True

    def turn_off(self):
        print("turning off the aircondition")
        self.is_on = False


class Heating(object):

    def __init__(self):
        self.is_on = False

    def __str__(self):
        return "on" if self.is_on else "off"

    def turn_on(self):
        print("turning on the heating")
        self.is_on = True

    def turn_off(self):
        print("turning off the heating")
        self.is_on = False


class Boiler(object):

    def __init__(self):
        self.temperature = 83

    def __str__(self):
        return "boiler temperature:{}".format(self.temperature)

    def increase_temperature(self, amount):
        print("increasing the boiler's temperature by {} degrees".format(amount))
        self.temperature += amount

    def decrease_temperature(self, amount):
        print("decreasing the boiler's temperature by {} degrees".format(amount))
        self.temperature -= amount


class Fridge(object):

    def __init__(self):
        self.temperature = 2

    def __str__(self):
        return "fridge temperature: {}".format(self.temperature)

    def increase_temperature(self, amount):
        print("increasing the fridge's temperature by {} degrees".format(amount))
        self.temperature += amount

    def decrease_temperature(self, amount):
        print("decreasing the fridge's temperature by {} degrees".format(amount))
        self.temperature -= amount


def main():
    pass


if __name__ == "__main__":
    main()







