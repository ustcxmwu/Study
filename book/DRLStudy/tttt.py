import multiprocessing
import time

class WorkerProcess(multiprocessing.Process):
    def __init__(self, process_id, register_event, register_queue):
        super().__init__()
        self.process_id = process_id
        self.register_event = register_event
        self.register_queue = register_queue

    def run(self):
        # 从 Manager 中获取全局的 Manager 和 Queue
        manager = multiprocessing.Manager()
        self.queue = manager.Queue()
        # 注册自己到主进程的 register_queue
        self.register_queue.put((self.process_id, self.queue))
        self.register_event.set()  # 通知主进程注册完成

        while True:
            message = self.queue.get()
            if message == "exit":
                print(f"Worker {self.process_id} exiting.")
                break
            print(f"Worker {self.process_id} received message: {message}")
            time.sleep(1)
            print(f"Worker {self.process_id} processed message: {message}")

class ProcessManager(multiprocessing.Process):
    def __init__(self, command_queue, register_queue, shared_dict):
        super().__init__()
        self.command_queue = command_queue
        self.register_queue = register_queue
        self.queues = shared_dict

    def run(self):
        while True:
            try:
                while not self.register_queue.empty():
                    process_id, queue = self.register_queue.get()
                    self.queues[process_id] = queue
                    print(f'Registered worker process {process_id}')

                while not self.command_queue.empty():
                    command = self.command_queue.get_nowait()
                    if command["action"] == "send_message":
                        self.send_message(command["process_id"], command["message"])
                    elif command["action"] == "terminate":
                        self.terminate()
                        return
            except multiprocessing.queues.Empty:
                time.sleep(0.1)  # 防止忙等待

    def send_message(self, process_id, message):
        if process_id in self.queues:
            self.queues[process_id].put(message)
        else:
            print(f'Invalid process_id: {process_id}')

    def terminate(self):
        for queue in self.queues.values():
            queue.put("exit")


if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')  # 确保使用 spawn 模式

    # 创建 Manager 并通过 Manager 创建共享数据结构
    manager = multiprocessing.Manager()
    command_queue = manager.Queue()
    register_queue = manager.Queue()
    shared_dict = manager.dict()

    # 启动管理进程
    process_manager = ProcessManager(command_queue, register_queue, shared_dict)
    process_manager.start()

    worker_processes = []
    for i in range(3):
        queue_event = multiprocessing.Event()
        worker = WorkerProcess(i, queue_event, register_queue)
        worker.start()
        queue_event.wait()  # 等待 Worker 注册完成
        worker_processes.append(worker)

    # 发送消息到 Worker 进程
    command_queue.put({"action": "send_message", "process_id": 0, "message": "Hello Worker 0"})
    command_queue.put({"action": "send_message", "process_id": 1, "message": "Hello Worker 1"})
    command_queue.put({"action": "send_message", "process_id": 2, "message": "Hello Worker 2"})

    # 给予一些时间处理消息
    time.sleep(5)

    # 发送终止命令
    command_queue.put({"action": "terminate"})

    # 等待管理进程结束
    process_manager.join()

    # 等待所有 Worker 进程结束
    for worker in worker_processes:
        worker.join()