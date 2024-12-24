import multiprocessing
import time

class WorkerProcess(multiprocessing.Process):
    def __init__(self, process_id, register_event, manager_queue):
        super().__init__()
        self.process_id = process_id
        self.register_event = register_event
        self.manager_queue = manager_queue

    def run(self):
        # Create a queue for this worker using Manager's Queue
        self.queue = multiprocessing.Manager().Queue()
        # Register the worker with the manager
        self.manager_queue.put((self.process_id, self.queue))
        self.register_event.set()  # Signal that registration is done

        while True:
            message = self.queue.get()
            if message == "exit":
                print(f"Process {self.process_id} exiting.")
                break
            print(f"Process {self.process_id} received message: {message}")
            # Simulate work
            time.sleep(1)
            print(f"Process {self.process_id} completed handling message: {message}")

class ProcessManager(multiprocessing.Process):
    def __init__(self, command_queue, register_queue):
        super().__init__()
        self.manager = multiprocessing.Manager()
        self.queues = self.manager.dict()
        self.command_queue = command_queue
        self.register_queue = register_queue

    def run(self):
        while True:
            # Check for new registrations
            while not self.register_queue.empty():
                process_id, queue = self.register_queue.get()
                self.register_process(process_id, queue)

            # Check for commands
            while not self.command_queue.empty():
                command = self.command_queue.get()
                if command["action"] == "send_message":
                    self.send_message(command["process_id"], command["message"])
                elif command["action"] == "terminate":
                    self.terminate()
                    return

            time.sleep(0.1)  # Avoid busy-wait loop

    def register_process(self, process_id, queue):
        self.queues[process_id] = queue
        print(f"Process {process_id} registered.")

    def send_message(self, process_id, message):
        if process_id in self.queues:
            self.queues[process_id].put(message)
        else:
            print(f"Invalid process_id: {process_id}")

    def terminate(self):
        for queue in self.queues.values():
            queue.put("exit")
        self.queues.clear()

if __name__ == "__main__":
    # multiprocessing.set_start_method("fork", force=True)
    manager = multiprocessing.Manager()
    command_queue = manager.Queue()
    register_queue = manager.Queue()

    process_manager = ProcessManager(command_queue, register_queue)
    process_manager.start()

    worker_processes = []
    for i in range(3):
        register_event = multiprocessing.Event()
        worker = WorkerProcess(i, register_event, register_queue)
        worker.start()
        register_event.wait()  # Ensure the worker process is registered
        worker_processes.append(worker)

    # Send commands to the manager to send messages to worker processes
    command_queue.put({"action": "send_message", "process_id": 0, "message": "Hello Process 0"})
    command_queue.put({"action": "send_message", "process_id": 1, "message": "Hello Process 1"})
    command_queue.put({"action": "send_message", "process_id": 2, "message": "Hello Process 2"})

    # Allow some time for processing
    time.sleep(5)

    # Send terminate command to the manager
    command_queue.put({"action": "terminate"})

    # Wait for manager to terminate
    process_manager.join()

    # Wait for all worker processes to terminate
    for worker in worker_processes:
        worker.join()