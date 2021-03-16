from abc import ABCMeta, abstractmethod


class State(metaclass=ABCMeta):

    @abstractmethod
    def write_program(self, w):
        pass


class ForenoonState(State):

    def write_program(self, w):
        if w.hour < 12:
            print("当前时间:{}, 精神百倍".format(w.hour))
        else:
            w.set_state(AfternoonState())
            w.write_program()


class AfternoonState(State):

    def write_program(self, w):
        if w.hour < 17:
            print("当前时间:{}, 状态还行, 继续努力".format(w.hour))
        else:
            w.set_state(EveningState())
            w.write_program()


class EveningState(State):

    def write_program(self, w):
        if w.hour < 21:
            print("当前时间:{}, 加班呢, 疲劳了".format(w.hour))
        else:
            w.set_state(SleepState())
            w.write_program()


class SleepState(State):

    def write_program(self, w):
        print("当前时间:{}, 睡觉了".format(w.hour))


class Work(object):

    def __init__(self):
        self.hour = 9
        self.curr = ForenoonState()

    def set_state(self, s: State):
        self.curr = s

    def write_program(self):
        self.curr.write_program(self)


def main():
    work = Work()
    work.hour = 9
    work.write_program()
    work.hour = 15
    work.write_program()
    work.hour = 20
    work.write_program()
    work.hour = 22
    work.write_program()


if __name__ == "__main__":
    main()
