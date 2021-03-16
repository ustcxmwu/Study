from enum import Enum
from abc import ABCMeta, abstractmethod

State = Enum("State", "new running sleeping restart zombie")


class Server(metaclass=ABCMeta):

    @abstractmethod
    def __init__(self):
        pass

    def __str__(self):
        return self.name

    @abstractmethod
    def boot(self):
        pass

    @abstractmethod
    def kill(self, restart=True):
        pass


class FileServer(Server):

    def __init__(self):
        self.name = "FileServer"
        self.state = State.new

    def boot(self):
        print("booting the {}".format(self))
        self.state = State.running

    def kill(self, restart=True):
        print("Killing {}".format(self))
        self.state = State.restart if restart else State.zombie

    def creat_file(self, user, name, permissions):
        print("trying to create the file '{}' for user '{}' with permissions {}".format(name, user, permissions))


class ProcessServer(Server):

    def __init__(self):
        self.name = "ProcessServer"
        self.state = State.new

    def boot(self):
        print("booting the {}".format(self))
        self.state = State.running

    def kill(self, restart=True):
        print("Killing {}".format(self))
        self.state = State.restart if restart else State.zombie

    def create_process(self, user, name):
        print("trying to create the process '{}' for user '{}'".format(name, user))


class OperationSystem(object):

    def __init__(self):
        self.fs = FileServer()
        self.ps = ProcessServer()

    def start(self):
        [i.boot() for i in (self.fs, self.ps)]

    def create_file(self, user, name, permissions):
        return self.fs.creat_file(user, name, permissions)

    def create_process(self, user, name):
        return self.ps.create_process(user, name)


def main():
    os = OperationSystem()
    os.start()
    os.create_file("foo", "hello", "-rw-r-r")
    os.create_process("bar", "ls /tmp")


if __name__ == "__main__":
    main()




