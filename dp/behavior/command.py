#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.
import os

verbose = True


class RenameFile(object):

    def __init__(self, path_src, path_dest):
        self.src = path_src
        self.dest = path_dest

    def execute(self):
        if verbose:
            print("[renaming '{}' to '{}']".format(self.src, self.dest))
        os.rename(self.src, self.dest)

    def undo(self):
        if verbose:
            print("[renameing [{}' to '{}']".format(self.src, self.dest))
        os.rename(self.dest, self.src)


class CreateFile(object):

    def __init__(self, path, txt="Hello world.\n"):
        self.path, self.txt = path, txt

    def execute(self):
        if verbose:
            print("[creating file '{}']".format(self.path))
        with open(self.path, mode='w') as outfile:
            outfile.write(self.txt)

    def undo(self):
        delete_file(self.path)


class ReadFile(object):

    def __init__(self, path):
        self.path = path

    def execute(self):
        if verbose:
            print("[reading file '{}']".format(self.path))
        with open(self.path, mode='r') as infile:
            print(infile.read(), end=" ")


def delete_file(path):
    if verbose:
        print("deleting file '{}'".format(path))
    os.remove(path)


def main():
    ori_name, new_name = "file1", "file2"
    commands = []
    for cmd in CreateFile(ori_name), ReadFile(ori_name), RenameFile(ori_name, new_name):
        commands.append(cmd)

    [command.execute() for command in commands]

    answer = input("reverse the executed commands? [y/n]")

    if answer not in "yY":
        print(("the result is {}".format(new_name)))
        exit()

    for c in reversed(commands):
        try:
            c.undo()
        except AttributeError as e:
            pass


if __name__ == "__main__":
    main()
