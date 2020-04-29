import pyglet
from . import world, player

# creates and names the window.
window = pyglet.window.Window(600, 500)
window.set_caption("Dirt Version 1.0 Beta")

# creates the event loop CONTAINER
event_loop = pyglet.app.EventLoop()

# brings together all the other code from the other files in a neat & compact way.
currentWorld = world.World()


@window.event
def on_draw():
    window.clear()
    currentWorld.draw_world()
    currentWorld.drawn_world.draw()


# world.pictures.wood.blit(0,500) #uncomment this is you want to print wood to the window.


@event_loop.event
def on_window_close(window):
    event_loop.exit()


if __name__ == '__main__':
    pyglet.app.run()