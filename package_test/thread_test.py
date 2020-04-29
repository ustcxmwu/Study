from threading import Thread
import time

def sayHi(name, sleep):
    time.sleep(sleep)
    print('%s say hello' % name)


def test1():
    t = Thread(target=sayHi, args=('wuxiaomin', 3))
    t.start()
    print("main thread.")


class Say(Thread):
    def __init__(self, name, sleep):
        super().__init__()
        self.name = name
        self.sleep = sleep

    def run(self):
        time.sleep(self.sleep)
        print('%s say hello' % self.name)


def test2():
    t = Say('wu', 1)
    t.start()
    print('main thread.')


if __name__ == '__main__':
    test2()
