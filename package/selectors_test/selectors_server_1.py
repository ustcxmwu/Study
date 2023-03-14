#  Copyright (c) 2020. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

from socket import *
import selectors

sel = selectors.DefaultSelector()


def accept(server_fileobj, mask):
    conn, addr = server_fileobj.accept()
    sel.register(conn, selectors.EVENT_READ, read)
    return True


def read(conn, mask):
    try:
        data = conn.recv(1024).decode('utf-8')
        if data == 'quit':
            print("client quit")
            conn.send("shut down".encode("utf-8"))
            sel.unregister(conn)
            conn.close()
            return False
        elif data == 'start':
            conn.send("start".encode('utf-8'))
            return True
        else:
            conn.send(data.upper().encode('utf-8') + b'_SB')
            return True
    except Exception:
        print('xxxxx closing', conn)
        sel.unregister(conn)
        conn.close()
        return True


if __name__ == '__main__':
    server_fileobj = socket(AF_INET, SOCK_STREAM)
    server_fileobj.setsockopt(SOL_SOCKET, SO_REUSEADDR, 1)
    server_fileobj.bind(('0.0.0.0', 8088))
    server_fileobj.listen(5)
    server_fileobj.setblocking(False)  # 设置socket的接口为非阻塞
    sel.register(server_fileobj, selectors.EVENT_READ,
                 accept)  # 相当于网select的读列表里append了一个文件句柄server_fileobj,并且绑定了一个回调函数accept

    while True:
        events = sel.select()  # 检测所有的fileobj，是否有完成wait data的
        sht = False
        for sel_obj, mask in events:
            callback = sel_obj.data  # callback=accpet
            ret = callback(sel_obj.fileobj, mask)  # accpet(server_fileobj,1)
            if not ret:
                sel.close()
                sht = True
                break
        if sht:
            break

