


from abc import ABCMeta, abstractmethod


class Command(metaclass=ABCMeta):

    def __init__(self, recv):
        self.recv = recv

    @abstractmethod
    def execute(self):
        pass


class ConcreteCommand(Command):

    def __init__(self, recv):
        self.recv = recv

    def execute(self):
        self.recv.action()


class Receiver(object):

    def action(self):
        print("Receiver Action")


class Invoker(object):

    def command(self, cmd):
        self.cmd = cmd

    def execute(self):
        self.cmd.execute()


def t_simple_command():
    recv = Receiver()
    cmd = ConcreteCommand(recv)
    invoker = Invoker()
    invoker.command(cmd)
    invoker.execute()


class Order(metaclass=ABCMeta):

    @abstractmethod
    def execute(self):
        pass


class BuyStockOrder(Order):

    def __init__(self, stock):
        self.stock = stock

    def execute(self):
        self.stock.buy()


class SellStockOrder(Order):

    def __init__(self, stock):
        self.stock = stock

    def execute(self):
        self.stock.sell()


class StockTrade(object):

    def buy(self):
        print("You will buf stocks")

    def sell(self):
        print("You will sell stocks")


class Agent(object):

    def __init__(self):
        self.__order_queue = []

    def place_order(self, order):
        self.__order_queue.append(order)
        order.execute()


if __name__ == '__main__':
    stock = StockTrade()
    buy_order = BuyStockOrder(stock)
    sell_order = SellStockOrder(stock)

    agent = Agent()
    agent.place_order(buy_order)
    agent.place_order(sell_order)

