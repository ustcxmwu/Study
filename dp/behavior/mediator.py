from abc import ABC, abstractmethod


class Mediator(ABC):

    def __init__(self, comp, name):
        self.name = name
        self.comp = comp

    @abstractmethod
    def send(self, message, colleague):
        pass


class Colleague(ABC):
    mediator = None

    def __init__(self, mediator):
        self.mediator = mediator


class ConcreteColleagueA(Colleague):

    def zhaozu(self, name, address, area, price):
        msg = "你好，我是房东{}, 我的房子在{}, 面积是{}, 租金是{}".format(name, address, area, price)
        return msg


class ConcreteColleagueB(Colleague):

    def qiuzu(self, name, address, area, price):
        msg = "你好, 我是租客{}, 我的想租的房子大概在{}, 面积大概{}, 租金大概{}".format(name, address, area, price)
        return msg


class ConcreteMediator(Mediator):

    def intor_self(self):
        print("您好, 我是{}公司的{}, 以后我来为您找到合适的房子或合适的租客".format(self.comp, self.name))

    def send(self, name, message):
        if name == "fd":
            print("房东您好, 已经收到你的放租消息, 我马上联系租客")
            print("zk 您好, 房东的消息是:{}".format(message))
        else:
            print("租客你好, 已收到你的求租消息, 我马上联系房东")
            print("fd 你好, 租客的消息是:{}".format(message))


def main():
    mediator = ConcreteMediator("zy", "xf")
    mediator.intor_self()
    c1 = ConcreteColleagueA(mediator)
    fdmsg = c1.zhaozu("fd", "nanshan", "80", "7000")
    mediator.send("fd", fdmsg)
    print("".center(80, "="))

    c2 = ConcreteColleagueB(mediator)
    zkmsg = c2.qiuzu("zk", "nanshan", "70", "6000")
    mediator.send("zk", zkmsg)



if __name__ == "__main__":
    main()


