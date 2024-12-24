import multiprocessing
import time

def worker(process_id, register_queue, register_event):
    manager = multiprocessing.Manager()
    queue = manager.Queue()

    # Register the process with the manager
    register_queue.put((process_id, queue))
    register_event.set()  # Signal the main process that registration is done

    while True:
        message = queue.get()
        if message == "exit":
            print(f"Process {process_id} exiting.")
            break
        print(f"Process {process_id} received message: {message}")
        # Simulate work
        time.sleep(1)
        print(f"Process {process_id} completed handling message: {message}")

def process_manager(command_queue, register_queue):
    manager = multiprocessing.Manager()
    queues = manager.dict()

    while True:
        while not register_queue.empty():
            process_id, queue = register_queue.get()
            queues[process_id] = queue
            print(f"Process {process_id} registered.")

        while not command_queue.empty():
            command = command_queue.get()
            if command["action"] == "send_message":
                process_id = command["process_id"]
                message = command["message"]
                if process_id in queues:
                    queues[process_id].put(message)
                else:
                    print(f"Invalid process_id: {process_id}")
            elif command["action"] == "terminate":
                for queue in queues.values():
                    queue.put("exit")
                return
        time.sleep(0.1)

if __name__ == '__main__':
    multiprocessing.set_start_method('spawn')

    manager = multiprocessing.Manager()
    command_queue = manager.Queue()
    register_queue = manager.Queue()

    # Start the process manager
    manager_process = multiprocessing.Process(target=process_manager, args=(command_queue, register_queue))
    manager_process.start()

    worker_processes = []
    for i in range(3):
        register_event = multiprocessing.Event()
        worker_process = multiprocessing.Process(target=worker, args=(i, register_queue, register_event))
        worker_process.start()
        register_event.wait()  # Wait until the worker process is registered
        worker_processes.append(worker_process)

    # Send commands to the manager to send messages to worker processes
    command_queue.put({"action": "send_message", "process_id": 0, "message": "Hello Process 0"})
    command_queue.put({"action": "send_message", "process_id": 1, "message": "Hello Process 1"})
    command_queue.put({"action": "send_message", "process_id": 2, "message": "Hello Process 2"})

    # Allow some time for processing
    time.sleep(5)

    # Send terminate command to the manager
    command_queue.put({"action": "terminate"})

    # Wait for manager to terminate
    manager_process.join()

    # Wait for all worker processes to terminate
    for worker_process in worker_processes:
        worker_process.join()