import queue
import threading
import operator


class Job(object):
    def __init__(self, priority, description):
        self.priority = priority
        self.description = description
        return

    def __lt__(self, other):
        return self.priority < other.priority


def process_job(q):
    while True:
        next_job = q.get()
        print("process job: {}".format(next_job.description))
        q.task_done()


def main():
    q = queue.PriorityQueue()
    q.put(Job(3, "job 3"))
    q.put(Job(10, "job 10"))
    q.put(Job(1, "job 1"))
    workers = [threading.Thread(target=process_job, args=(q,)), threading.Thread(target=process_job, args=(q,))]

    for w in workers:
        w.setDaemon(True)
        w.start()
    q.join()


if __name__ == "__main__":
    main()
