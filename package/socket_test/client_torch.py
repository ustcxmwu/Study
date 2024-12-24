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


# 初始化Socket客户端
def start_client():
    client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_host = "localhost"
    server_port = 65432
    client_socket.connect((server_host, server_port))

    # 创建一个示例模型并初始化
    model = SimpleModel()
    model.fc.weight.data.fill_(1.0)
    model.fc.bias.data.fill_(0.5)

    # 序列化模型
    buffer = io.BytesIO()
    torch.save(model, buffer)
    buffer.seek(0)
    data = buffer.read()

    # 发送模型数据
    client_socket.sendall(data)
    print("Model sent.")

    # 关闭连接
    client_socket.close()


if __name__ == "__main__":
    start_client()
