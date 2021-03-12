#  Copyright (c) 2021. Xiaomin Wu <xmwu@mail.ustc.edu.cn>
#  All rights reserved.

class Frog(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def interact_with(self, obstacle):
        print("{} the Frog encounters {} and {}!".format(self, obstacle, obstacle.action()))


class Bug(object):

    def __str__(self):
        return "a bug"

    def action(self):
        return "eats it"


class FrogWorld(object):

    def __init__(self, name):
        print(self)
        self.player_name = name

    def __str__(self):
        return "\n\n\t ---------------------Frog World -----------------"

    def make_character(self):
        return Frog(self.player_name)

    def make_obstacle(self):
        return Bug()


class Wizard(object):

    def __init__(self, name):
        self.name = name

    def __str__(self):
        return self.name

    def interact_with(self, obstacle):
        print("{} the Wizard battles against {} and {}".format(self, obstacle, obstacle.action()))


class Ork(object):

    def __str__(self):
        return "an evil ork"

    def action(self):
        return "kills it"


class WizardWorld(object):

    def __init__(self, name):
        print(self)
        self.player_name = name

    def __str__(self):
        return "\n\n\n ----------------------Wizard World ----------------------------------"

    def make_character(self):
        return Wizard(self.player_name)

    def make_obstacle(self):
        return Ork()


class GameEnvironment(object):

    def __init__(self, factory):
        self.hero = factory.make_character()
        self.obstacle = factory.make_obstacle()

    def play(self):
        self.hero.interact_with(self.obstacle)


def validate_age(name):
    try:
        age = input("Welcome {}, how old are you?".format(name))
        age = int(age)
    except ValueError as ve:
        print("Age {} is invalid, please try again...".format(age))
        return (False, age)
    return (True, age)


def main():
    name = input("Hello, what is your name?")
    valid_input = False
    while not valid_input:
        valid_input, age = validate_age(name)
    game = FrogWorld if age < 18 else WizardWorld
    env = GameEnvironment(game(name))
    env.play()


if __name__ == "__main__":
    main()





