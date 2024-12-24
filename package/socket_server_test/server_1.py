import socketserver


# 定义请求处理器类，继承自socketserver.BaseRequestHandler
class ThreadedTCPRequestHandler(socketserver.BaseRequestHandler):
    def handle(self):
        # self.request是客户端的socket连接
        data = self.request.recv(1024).strip()
        print(f"Received from {self.client_address}: {data.decode('utf-8')}")

        # 发送响应到客户端
        response = f"Server received: {data.decode('utf-8')}"
        self.request.sendall(response.encode("utf-8"))


# 定义多线程TCP服务器类，继承自socketserver.ThreadingMixIn和socketserver.TCPServer
class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
    pass


# 创建并启动服务器
if __name__ == "__main__":
    HOST, PORT = "localhost", 65432

    server = ThreadedTCPServer((HOST, PORT), ThreadedTCPRequestHandler)
    with server:
        print(f"Server started at {HOST}:{PORT}")
        server.serve_forever()
