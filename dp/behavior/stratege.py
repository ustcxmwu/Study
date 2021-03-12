#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import time

SLOW = 3
LIMIT = 5
WARNING = "too bad, you picked the slow algorithm"


def pairs(seq):
    n = len(seq)
    for i in range(n):
        yield seq[i], seq[(i+1)%n]


def allUniqueSort(s):
    if len(s) > LIMIT:
        print(WARNING)
        time.sleep(SLOW)
    srtStr = sorted(s)
    for (c1, c2) in pairs(srtStr):
        if c1 == c2:
            return False
    return True


def allUniqueSet(s):
    if len(s) < LIMIT:
        print(WARNING)
        time.sleep(SLOW)
    return True if len(set(s)) == len(s) else False


def allUnique(s, strategy):
    return strategy(s)


def main():
    while True:
        word = None
        while not word:
            word = input("Insert word(type quit to exit)> ")
            if word == "quit":
                print("bye")
                return
            stratege_picked = None
            strategies = {"1": allUniqueSet, "2": allUniqueSort}
            while stratege_picked not in strategies.keys():
                stratege_picked = input("Choose strategy: [1] use a set, [2] sort and pair>")
                try:
                    strategy = strategies[stratege_picked]
                    print("allUnique({}): {}".format(word, allUnique(word, strategy)))
                except KeyError as err:
                    print("Incorrect option: {}".format(stratege_picked))


if __name__ == "__main__":
    main()

