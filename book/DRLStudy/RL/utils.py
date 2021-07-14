from contextlib import contextmanager
import time


@contextmanager
def timer(name):
    start = time.time()
    yield
    end = time.time()
    print("{} Cost:{}".format(name, end-start))