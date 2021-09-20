import pymunk
from pyglet import *
import pymunk.pyglet_util

space = pymunk.Space()
body = pymunk.Body(1, 10)
shape = pymunk.Circle(body, 10)
shape.color = (255, 0, 0, 255) # will draw my_shape in red
space.add(body, shape)
win = window.Window(600, 600, "test")
app.run()
options = pymunk.pyglet_util.DrawOptions()
space.debug_draw(options)