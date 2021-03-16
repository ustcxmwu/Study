import time
from abc import ABCMeta, abstractmethod


SLOW = 3
LIMIT = 5
WARNING = "too bad, you picked the slow algorithm"


class Strategy(metaclass=ABCMeta):

    @abstractmethod
    def unique(self, s):
        pass


class SetUniqueStrategy(Strategy):

    def unique(self, s):
        if len(s) < LIMIT:
            print(WARNING)
            time.sleep(SLOW)
        return True if len(set(s)) == len(s) else False


class SortPairUniqueStragety(Strategy):

    def pairs(self, seq):
        n = len(seq)
        for i in range(n):
            yield seq[i], seq[(i + 1) % n]

    def unique(self, s):
        if len(s) > LIMIT:
            print(WARNING)
            time.sleep(SLOW)
        srtStr = sorted(s)
        for (c1, c2) in self.pairs(srtStr):
            if c1 == c2:
                return False
        return True


def allUnique(s, strategy: Strategy):
    return strategy.unique(s)


def main():
    while True:
        word = None
        while not word:
            word = input("Insert word(type quit to exit)> ")
            if word == "quit":
                print("bye")
                return
            stratege_picked = None
            strategies = {"1": SetUniqueStrategy, "2": SortPairUniqueStragety}
            while stratege_picked not in strategies.keys():
                stratege_picked = input("Choose strategy: [1] use a set, [2] sort and pair>")
                try:
                    strategy = strategies[stratege_picked]
                    print("allUnique({}): {}".format(word, allUnique(word, strategy())))
                except KeyError as err:
                    print("Incorrect option: {}".format(stratege_picked))


if __name__ == "__main__":
    main()
