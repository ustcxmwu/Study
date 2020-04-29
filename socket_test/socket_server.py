import socketserver

ip_port = ("127.0.0.1", 8000)


class MyServer(socketserver.BaseRequestHandler):

    def handle(self):
        print("conn is :", self.request)  # conn
        print("addr is :", self.client_address)  # addr

        while True:
            try:
                # 收消息
                data = self.request.recv(1024)
                if not data:
                    break
                print("收到客户端的消息是", data.decode("utf-8"))
                # 发消息
                self.request.sendall(data.upper())
            except Exception as e:
                print(e)
                break


if __name__ == "__main__":
    s = socketserver.ThreadingTCPServer(ip_port, MyServer)
    s.serve_forever()
