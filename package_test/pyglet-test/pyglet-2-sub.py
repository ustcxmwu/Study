import pyglet
import numpy as np
from PIL import Image
import copy


def img_covert_pyglet_pil(pyglet_img):
    data = pyglet_img.get_image_data()
    pil_img = Image.frombytes('RGBA', (pyglet_img.width, pyglet_img.height), data.get_data('RGBA', data.pitch)).transpose(Image.FLIP_TOP_BOTTOM)
    # pil_img = Image.frombytes('RGBA', (pyglet_img.width, pyglet_img.height), data.get_data('RGBA', data.pitch))
    # pil_img.show()
    return pil_img


def capture(self):
    pyglet_img = copy.deepcopy(pyglet.image.get_buffer_manager().get_color_buffer())
    img = img_covert_pyglet_pil(pyglet_img)
    return img


class Env(pyglet.window.Window):

    def __init__(self, width=500, height=500, visible=True):
        super(Env, self).__init__(width, height, visible=visible)
        self.acvive = 1
        self.batch = pyglet.graphics.Batch()
        self.background = pyglet.graphics.OrderedGroup(0)
        self.foreground = pyglet.graphics.OrderedGroup(1)

        self.obstacle_coords = np.array([
            [180, 120],
            [320, 120],
            [380, 180],
            [380, 320],
            [320, 380],
            [180, 380],
            [120, 320],
            [120, 180],
        ])
        self.batch.add(8, pyglet.gl.GL_POLYGON, self.foreground, ('v2f', self.obstacle_coords.flatten()),
                       ('c3B', (134, 181, 244) * 8))
        # pyglet.image.get_buffer_manager().get_color_buffer().('screenshot.png')

    def render(self):
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.clear()
        self.batch.draw()
        self.dispatch_events()
        pyglet_img = copy.deepcopy(pyglet.image.get_buffer_manager().get_color_buffer())
        img = img_covert_pyglet_pil(pyglet_img)
        self.flip()
        return img

    def on_draw(self):
        self.render()

    def on_close(self):
        self.alive = 0

    def run(self):
        # while self.acvive == 1:
        self.render()
        event = self.dispatch_events()


if __name__ == '__main__':
    env = Env()
    env.run()
    img_data = env.render()

    img_data.show()
    # img = Image.fromarray(img_data)
    # img_data.save('test.png')
