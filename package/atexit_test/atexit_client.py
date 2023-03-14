import atexit


@atexit.register
def goodbye():
    print("goodbye")


@atexit.register
def goodbye2():
    print("gggggggggggggggggggggggggggggggggggggg")


def main():
    pass


if __name__ == "__main__":
    main()
