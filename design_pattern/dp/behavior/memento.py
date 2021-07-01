class Originator(object):

    def __init__(self, state):
        self.state = state

    def create_memento(self):
        return Memento(self.state)

    def set_memento(self, memento):
        self.state = memento.state

    def show(self):
        print("当前状态: {}".format(self.state))


class Memento(object):

    def __init__(self, state):
        self.state = state


class Caretaker(object):
    def __init__(self, memento):
        self.memento = memento


def main():
    originator = Originator(state="On")
    originator.show()

    caretaker = Caretaker(originator.create_memento())
    originator.state = "Off"
    originator.show()

    originator.set_memento(caretaker.memento)
    originator.show()


if __name__ == "__main__":
    main()

