import queue
import threading
from typing import NamedTuple



def func1():
    q = queue.Queue()
    for i in range(5):
        q.put(i)

    while not q.empty():
        print(q.get())


def func2():
    q = queue.LifoQueue()
    for i in range(5):
        q.put(i)

    while not q.empty():
        print(q.get())


class Job(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        print("Job:", description)

    def __lt__(self, other):
        return self.priority < other.priority


def func3():
    q = queue.PriorityQueue()
    q.put(Job(3, 'level 3'))
    q.put(Job(10, 'level 10'))
    q.put(Job(1, 'level 1'))

    def process_job(q):
        while True:
            next_job = q.get()
            print('for:', next_job.description)
            q.task_done()

    workers = [threading.Thread(target=process_job, args=(q,)), threading.Thread(target=process_job, args=(q,))]
    for w in workers:
        w.setDaemon(True)
        w.start()

    q.join()

def func4():
    Friend = NamedTuple("Friend", ['name', 'age', 'email'])
    f1 = Friend('xiaowang', 33, 'xiaowang@163.com')
    print(f1)
    print(f1.age)
    print(f1.email)
    f2 = Friend(name='xiaozhang', email='xiaozhang@sina.com', age=30)
    print(f2)
    name, age, email = f2
    print(name, age, email)


if __name__ == '__main__':
    func4()