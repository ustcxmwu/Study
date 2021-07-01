from cowpy import cow
from abc import ABCMeta, abstractmethod


def dots_style(msg):
    msg = msg.capitalize()
    msg = msg.center(80, ".")
    return msg


def admire_style(msg):
    msg = msg.upper()
    return "!".join(msg)


def cow_style(msg):
    msg = cow.milk_random_cow(msg)
    return msg


def generate_banner(msg, style=dots_style):
    print("-- start of banner --")
    print(style(msg))
    print("-- end of banner --\n\n")


def main():
    msg = "happy coding"
    [generate_banner(msg, style) for style in (dots_style, admire_style, cow_style)]


if __name__ == "__main__":
    main()