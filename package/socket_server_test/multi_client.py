import threading
import socket


def start_client(client_id):
    HOST, PORT = "localhost", 65432

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        # 连接到服务器
        sock.connect((HOST, PORT))

        # 发送数据到服务器
        message = f"Hello, Server! This is client {client_id}"
        print(f"Client {client_id} Sending: {message}")
        sock.sendall(message.encode("utf-8"))

        # 接收服务器的响应
        response = sock.recv(1024)
        print(f"Client {client_id} Received: {response.decode('utf-8')}")


if __name__ == "__main__":
    threads = []
    num_clients = 5
    for i in range(num_clients):
        t = threading.Thread(target=start_client, args=(i,))
        t.start()
        threads.append(t)

    for t in threads:
        t.join()
