# -*- coding: utf-8 -*-

import subprocess


def main():
    child = subprocess.Popen('ping baidu.com', shell=True)
    child.wait()



if __name__ == "__main__":
    main()