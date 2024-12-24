import socket
import torch
import torch.nn as nn
import io


# 定义一个简单的示例模型
class SimpleModel(nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.fc = nn.Linear(10, 1)

    def forward(self, x):
        return self.fc(x)


# 初始化Socket服务器
def start_server():
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_host = "localhost"
    server_port = 65432
    server_socket.bind((server_host, server_port))
    server_socket.listen(1)

    print(f"Server started at {server_host}:{server_port}")

    while True:
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")

        # 接收数据并反序列化模型
        data = b""
        while True:
            packet = client_socket.recv(4096)
            if not packet:
                break
            data += packet

        # 反序列化模型
        buffer = io.BytesIO(data)
        model = torch.load(buffer)
        print("Model received and loaded.")

        # 显示模型参数
        for param_tensor in model.state_dict():
            print(param_tensor, "\t", model.state_dict()[param_tensor].size())

        # 关闭客户端连接
        client_socket.close()


if __name__ == "__main__":
    start_server()
