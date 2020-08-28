import platform


if __name__ == '__main__':
    print(platform.system())
    print(platform.architecture())
    print(platform.node())
    print(platform.uname())
    print(platform.python_version())