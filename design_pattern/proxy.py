from abc import ABCMeta, abstractmethod


class Actor(object):

    def __init__(self):
        self.is_busy = False

    def occupied(self):
        self.is_busy = True
        print(type(self).__name__, "is occupied with current movie")

    def available(self):
        self.is_busy = False
        print(type(self).__name__, "is free for the movie")

    def get_status(self):
        return self.is_busy


class Agent(object):

    def __init__(self):
        self.principal = None

    def work(self):
        self.actor = Actor()
        if self.actor.get_status():
            self.actor.occupied()
        else:
            self.actor.available()


def test_simple_agent():
    r = Agent()
    r.work()


class You(object):

    def __init__(self):
        print("You:: Lets buy the Denim shirt")
        self.debit_card = DebitCard()
        self.is_purchased = None

    def make_payment(self):
        self.is_purchased = self.debit_card.do_pay()

    def __del__(self):
        if self.is_purchased:
            print("You:: Wow! Denim shirt is Mine :-)")
        else:
            print("You:: I should earn more :(")


class Payment(metaclass=ABCMeta):

    @abstractmethod
    def do_pay(self):
        pass


class Bank(Payment):

    def __init__(self):
        self.card = None
        self.account = None

    def __get_account(self):
        self.account = self.card
        return self.account

    def __has_funds(self):
        print("Bank:: Checking if Account", self.__get_account(), "has enough money")
        return True

    def set_card(self, card):
        self.card = card

    def do_pay(self):
        if self.__has_funds():
            print("Bank:: Paying the merchant")
            return True
        else:
            print("Bank:: Sorry, not enough funds")
            return False


class DebitCard(Payment):

    def __init__(self):
        self.bank = Bank()

    def do_pay(self):
        card = input("Proxy:: Punch in Card Number:")
        self.bank.set_card(card)
        return self.bank.do_pay()


def test_payment():
    you = You()
    you.make_payment()


if __name__ == '__main__':
    test_payment()
