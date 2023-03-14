import logging
import random
import time
from multiprocessing import Process

import zmq
from zmq.log.handlers import PUBHandler


class LogPublisher(object):

    def __init__(self, ip = "127.0.0.1", port = 8000):
        self._logger = logging.getLogger(__name__)
        self._logger.setLevel(logging.DEBUG)
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.PUB)
        self.socket.bind("tcp://{}:{}".format(ip, port))
        self.handler = PUBHandler(self.socket)
        self.format = logging.Formatter("[%(filename)s:%(lineno)d] %(asctime)s %(levelname)s %(message)s")
        self.handler.setFormatter(self.format)
        self._logger.addHandler(self.handler)

    @property
    def logger(self):
        return self._logger


class Publisher(Process):

    def __init__(self):
        super().__init__()
        self.logger = None

    def run(self):
        self.logger = LogPublisher().logger
        while True:
            topic = random.randrange(9999, 10005)
            msg = random.randrange(1, 215) - 80
            print("before publisher logging.")
            self.logger.info("{} {}".format(topic, msg))
            time.sleep(1)


if __name__ == '__main__':
    pub = Publisher()
    pub.start()
    pub.join()







