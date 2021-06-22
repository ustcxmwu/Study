import pyglet


def test_window():
    game_window = pyglet.window.Window(800, 600)
    pyglet.app.run()

window = pyglet.window.Window(800, 600)
score_label = pyglet.text.Label(text="Score: 0", x=10, y=575)
level_label = pyglet.text.Label(text="My Amazing Game", x=400, y=575, anchor_x='center')


@window.event
def on_draw():
    pyglet.resource.path = ['./resouce']
    pyglet.resource.reindex()
    image1 = pyglet.resource.image('1.png')
    image2 = pyglet.resource.image('2.png')
    window.clear()
    image1.blit(0, 0)
    score_label.draw()
    level_label.draw()


def center_image(image):
    """Sets an image's anchor point to its center"""
    image.anchor_x = image.width/2
    image.anchor_y = image.height/2





if __name__ == '__main__':
    # pyglet.app.run()
    pass
