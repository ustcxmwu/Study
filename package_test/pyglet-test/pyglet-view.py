import numpy as np
import pyglet
import PIL


pyglet.clock.set_fps_limit(10000)


class Viewer(pyglet.window.Window):
    color = {
        'background': [1]*3 + [1]
    }
    fps_display = pyglet.clock.ClockDisplay()
    bar_thc = 5

    def __init__(self, win, obstacle, hero, enemy):
        super(Viewer, self).__init__(win[0], win[1], display=False)
        self.set_location(x=80, y=10)
        pyglet.gl.glClearColor(*self.color['background'])

        self.hero = hero
        self.obstacle = obstacle

        self.batch = pyglet.graphics.Batch()
        background = pyglet.graphics.OrderedGroup(0)
        foreground = pyglet.graphics.OrderedGroup(1)

        line_coord = [0, 0] * 2
        c = (73, 73, 73) * 2
        for i in range(len(self.sensor_info)):
            self.sensors.append(self.batch.add(2, pyglet.gl.GL_LINES, foreground, ('v2f', line_coord), ('c3B', c)))

        car_box = [0, 0] * 4
        c = (249, 86, 86) * 4
        self.car = self.batch.add(4, pyglet.gl.GL_QUADS, foreground, ('v2f', car_box), ('c3B', c))

        # c = (134, 181, 244) * 4
        # self.obstacle = self.batch.add(4, pyglet.gl.GL_QUADS, background, ('v2f', obstacle_coords.flatten()), ('c3B', c))
        c = (134, 181, 244) * 8
        self.obstacle = self.batch.add(8, pyglet.gl.GL_POLYGON, background, ('v2f', obstacle_coords.flatten()), ('c3B', c))

    def render(self):
        pyglet.clock.tick()
        self._update()
        self.switch_to()
        self.dispatch_events()
        self.dispatch_event('on_draw')
        self.flip()



if __name__ == '__main__':
    window = pyglet.window.Window(500, 500)
    
