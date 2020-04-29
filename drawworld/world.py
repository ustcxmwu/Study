# imports and sets up all the file dependancies
import pyglet
from random import *
from game import resources

pictures = resources.Pictures()


class World:
    def __init__(self):
        # prepares the world for being drawn
        self.drawn_world = pyglet.graphics.Batch()
        self.world = []
        self.world_length = 40

        # sets up the world
        self.top = []  # heaven
        self.common = []  # earth
        self.cave = []  # caves
        self.bottom = []  # underworld

        # creates the world
        self.top, self.bottom = self.make_rows(0, 0, 15), self.make_rows(0, 0, 25)
        self.common = self.make_rows(0, 0, 4) + self.make_rows(0, 1, 6)
        self.cave = self.make_rows(0, 4, 20)

        # creates grass
        self.create_grass(self.common)
        self.create_grass(self.cave, 20)
        self.load_world(0, 0)

    def make_rows(self, bottom_number=0, top_number=2, depth=10):
        return [[randint(bottom_number, top_number) for x in range(self.world_length)] for y in range(depth)]

    def create_grass(self, array_name, depth=10):

        for y in range(depth):
            for x in range(self.world_length):
                if x < 1:
                    continue
                elif array_name[y][x - 1] == 0 and array_name[y][x] == 1:
                    array_name[y][x] = 2
                elif (not array_name[y][x - 1] == 0) and array_name[y][x] == 2:
                    array_name[y][x] = 1

    def write_out_world(self):
        for x in self.top:
            for y in x:
                print(y, sep='', end=' ')
            print()
        print("---------------------------------------------------------------------------\n")
        for x in self.common:
            for y in x:
                print(y, sep='', end=' ')
            print()
        print("---------------------------------------------------------------------------\n")
        for x in self.cave:
            for y in x:
                print(y, sep='', end=' ')
            print()
        print("---------------------------------------------------------------------------\n")
        for x in self.bottom:
            for y in x:
                print(y, sep='', end=' ')
            print()
        print("---------------------------------------------------------------------------\n")

    def load_world(self, player_x, player_y):
        x_count = -1
        for x in self.top, self.common, self.cave, self.bottom:
            x_count += 1
            y_count = -1
            for y in x:
                y_count += 1
                if y == 0:
                    continue
                elif y == 1:
                    world.append(pyglet.sprite.Sprite(pictures.dirt, player_x + x_count * 10, player_y + y_count *
                                                         10,
                                                      batch=self.drawn_world))
                elif y == 2:
                    world.append(pyglet.sprite.Sprite(pictures.grass, player_x + x_count * 10, player_y + y_count * 10,
                                                      batch=self.drawn_world))
                elif y == 3:
                    world.append(pyglet.sprite.Sprite(pictures.stone, player_x + x_count * 10, player_y + y_count * 10,
                                                      batch=self.drawn_world))
                elif y == 4:
                    world.append(pyglet.sprite.Sprite(pictures.sand, player_x + x_count * 10, player_y + y_count * 10,
                                                      batch=self.drawn_world))

    def draw_world(self):
        self.drawn_world.draw()