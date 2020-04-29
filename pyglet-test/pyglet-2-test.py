import pyglet
import numpy as np
from PIL import Image
import copy
from pyglet.gl import *
import math

RAD2DEG = 57.29577951308232

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

class Attr(object):
    def enable(self):
        raise NotImplementedError

    def disable(self):
        pass

class Color(Attr):
    def __init__(self, vec4):
        self.vec4 = vec4

    def enable(self):
        glColor4f(*self.vec4)

class Geom(object):
    def __init__(self):
        self._color = Color((0, 0, 0, 1.0))
        self.attrs = [self._color]

    def render(self):
        for attr in reversed(self.attrs):
            attr.enable()
        self.render1()
        for attr in self.attrs:
            attr.disable()

    def render1(self):
        raise NotImplementedError

    def add_attr(self, attr):
        self.attrs.append(attr)

    def set_color(self, r, g, b, alpha=1):
        self._color.vec4 = (r, g, b, alpha)


class FilledPolygon(Geom):
    def __init__(self, v):
        Geom.__init__(self)
        self.v = v

    def render1(self):
        if len(self.v) == 4:
            glBegin(GL_QUADS)
        elif len(self.v) > 4:
            glBegin(GL_POLYGON)
        else:
            glBegin(GL_TRIANGLES)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()

        color = (
        self._color.vec4[0] * 0.5, self._color.vec4[1] * 0.5, self._color.vec4[2] * 0.5, self._color.vec4[3] * 0.5)
        glColor4f(*color)
        glBegin(GL_LINE_LOOP)
        for p in self.v:
            glVertex3f(p[0], p[1], 0)  # draw each vertex
        glEnd()


class Env:
    def __init__(self):
        self.window = pyglet.window.Window(width=500, height=500, visible=True)
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
        # self.window.on_close = self.window_closed_by_user()
        glLineWidth(2.0)

    def close(self):
        self.window.close()

    def window_closed_by_user(self):
        self.close()

    def render(self):
        pyglet.gl.glClearColor(1, 1, 1, 1)
        self.window.clear()
        self.window.switch_to()
        self.window.dispatch_events()
        poly = FilledPolygon(self.obstacle_coords)
        poly.render()
        pyglet_img = copy.deepcopy(pyglet.image.get_buffer_manager().get_color_buffer())
        img = img_covert_pyglet_pil(pyglet_img)
        self.window.flip()
        return img


if __name__ == '__main__':
    # env = Env()
    # img_data = env.render()
    # img_data.show()
    # img_data.save('test.png')
    # print(math.degrees(2.114))
    print(not (0 < 100 < 500 and 0 < 200 < 500))
    print(np.cos(np.pi))