import logging
from logging import handlers
from multiprocessing import Process

import zmq


class Subscriber(Process):

    def __init__(self, port):
        super().__init__()
        self.port = port

    def run(self):
        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect("tcp://localhost:{}".format(self.port))
        self.socket.setsockopt_string(zmq.SUBSCRIBE, '')
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.DEBUG)
        file_handler = handlers.RotatingFileHandler("./client_debug.log", maxBytes=10*1024*1024, backupCount=5)
        stream_handler = logging.StreamHandler()
        self.logger.addHandler(file_handler)
        self.logger.addHandler(stream_handler)
        while True:
            # topic, msg = self.socket.recv_multipart()
            topic, msg = self.socket.recv_multipart()
            self.logger.log(getattr(logging, topic.decode()), msg.decode().strip())



if __name__ == '__main__':
    sub = Subscriber(8000)
    sub.start()
    sub.join()







