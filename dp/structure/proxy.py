class SensitiveInfo(object):
    
    def __init__(self):
        self.users = ["nick", "tom", "ben", "mike"]

    def read(self):
        print("There are {} users:{}".format(len(self.users), " ".join(self.users)))

    def add(self, user):
        self.users.append(user)
        print("Add user {}".format(user))


class Info(object):

    def __init__(self):
        self.protected = SensitiveInfo()
        self.secret = "deadbeef"

    def read(self):
        self.protected.read()

    def add(self, user):
        sec = input("What is the secret?")
        self.protected.add(user) if sec == self.secret else print("That is Wrong.")


def main():
    info = Info()
    while True:
        print("1. read list |==| 2. add users |==| 3.quit")
        key = input("CHoose option:")
        if key ==  "1":
            info.read()
        elif key == "2":
            name = input("Choose username: ")
            info.add(name)
        elif key == "3":
            exit()
        else:
            print("Unknown option: {}".format(key))


if __name__ == "__main__":
    main()

        
