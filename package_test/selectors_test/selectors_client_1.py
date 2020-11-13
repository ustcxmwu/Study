
import socket
import select
import sys
from queue import Queue



if __name__ == '__main__':
    s = socket.socket()
    s.connect(('0.0.0.0', 8088))
    inputs = [s, sys.stdin]
    data_queue = Queue()
    while True:
        rs, ws, es = select.select(inputs, [], [], 0.1)
        for r in rs:
            if r is s:
                data = r.recv(1024).decode('utf-8')
                if data == "shut down":
                    print(data)
                    inputs.clear()
                    sys.exit(0)
                else:
                    print(data)
            elif r is sys.stdin:
                print("get_input")
                data = sys.stdin.readline().strip("\n")
                s.send(data.encode('utf-8'))