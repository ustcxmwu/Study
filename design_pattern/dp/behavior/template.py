from abc import ABCMeta, abstractmethod


class Drink(metaclass=ABCMeta):

    def heat_water(self):
        print("把水烧开")

    def pot(self):
        print("把水灌入壶中")

    @abstractmethod
    def add_condiments(self):
        raise NotImplementedError

    @abstractmethod
    def brew(self):
        raise NotImplementedError

    def drink(self):
        print("请慢用")


class Tea(Drink):

    def make(self):
        print("开始煮茶".center(80, "="))
        self.heat_water()
        self.pot()
        self.add_condiments()
        self.brew()
        self.drink()

    def add_condiments(self):
        print("放入茶叶")

    def brew(self):
        print("用煮茶的方式煮茶，倒掉第一泡的水")


class Coffee(Drink):

    def make(self):
        print("开始煮咖啡".center(80, "="))
        self.heat_water()
        self.pot()
        self.add_condiments()
        self.drink()

    def add_condiments(self):
        print("加入咖啡粉")

    def brew(self):
        print("依个人口味加入适量的糖和牛奶")


def main():
    t = Tea()
    t.make()
    c = Coffee()
    c.make()


if __name__ == "__main__":
    main()
