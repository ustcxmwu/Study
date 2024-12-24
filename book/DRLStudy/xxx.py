import multiprocessing
import time


class GrandchildProcess(multiprocessing.Process):
    def __init__(self, shared_queue, name):
        super().__init__()
        self.shared_queue = shared_queue
        self.name = name

    def run(self):
        for i in range(3):
            self.shared_queue.put(f"{self.name} - {i}")
            print(f"Grandchild {self.name} put: {self.name} - {i}")
            time.sleep(1)


class ChildProcess(multiprocessing.Process):
    def __init__(self, shared_queue, name):
        super().__init__()
        self.shared_queue = shared_queue
        self.name = name

    def run(self):
        # Create and start the grandchild process
        grandchild = GrandchildProcess(self.shared_queue, self.name)
        grandchild.start()

        # Continue child tasks, for example:
        for i in range(3):
            received_item = self.shared_queue.get()
            print(f"Child {self.name} got: {received_item}")

        # Wait for grandchild to finish
        grandchild.join()


class MainProcess:
    def __init__(self):
        self.shared_queue = multiprocessing.Queue()

    def start_processes(self):
        # Create child processes
        child1 = ChildProcess(self.shared_queue, "Child1")
        child2 = ChildProcess(self.shared_queue, "Child2")

        # Start child processes
        child1.start()
        child2.start()

        # Wait for child processes to finish
        child1.join()
        child2.join()

        print("All tasks are done.")


if __name__ == "__main__":
    main_process = MainProcess()
    main_process.start_processes()
