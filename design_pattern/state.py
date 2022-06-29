from abc import ABCMeta, abstractmethod


class State(metaclass=ABCMeta):

    @abstractmethod
    def handle(self):
        pass


class ConcreteStateA(State):

    def handle(self):
        print("ConcreteStateA")


class ConcreteStateB(State):

    def handle(self):
        print("ConcreteStateB")


class Context(State):

    def __init__(self):
        self._state = None

    @property
    def state(self):
        return self._state

    @state.setter
    def state(self, value):
        self._state = value

    def handle(self):
        self.state.handle()


if __name__ == '__main__':
    context = Context()
    stateA = ConcreteStateA()
    stateB = ConcreteStateB()

    context.state = stateA
    context.handle()
