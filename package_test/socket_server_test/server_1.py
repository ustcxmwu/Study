#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
from socketserver import StreamRequestHandler, ThreadingTCPServer


class ThreadedTCPRequestHandler(StreamRequestHandler):
    def handle(self):
        print(self.server.mycustomdata)



if __name__ == '__main__':

    server = ThreadingTCPServer(('127.0.0.1', 8000), ThreadedTCPRequestHandler)
    server.mycustomdata = 'foo.bar.z'
    server.serve_forever()

