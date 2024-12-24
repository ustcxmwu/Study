import socket

def start_client():
    HOST, PORT = "localhost", 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 连接到服务器
        sock.connect((HOST, PORT))

        # 发送数据到服务器
        message = "Hello, Server!"
        print(f"Sending: {message}")
        sock.sendall(message.encode('utf-8'))

        # 接收服务器的响应
        response = sock.recv(1024)
        print(f"Received: {response.decode('utf-8')}")

if __name__ == "__main__":
    start_client()