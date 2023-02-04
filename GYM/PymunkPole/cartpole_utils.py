import pygame
import pymunk
import sys
from pygame.locals import (QUIT, KEYDOWN, K_ESCAPE,K_LEFT,K_RIGHT)

##############################################################################
# Pygame
def handlePygameEvents(obj):
    obj.manual_force = 0
    for event in pygame.event.get():
        if event.type == QUIT:
            sys.exit(0)
        elif event.type == KEYDOWN and event.key == K_ESCAPE:
            sys.exit(0)
        elif event.type == KEYDOWN and event.key == K_LEFT:
            obj.manual_force=-100
        elif event.type == KEYDOWN and event.key == K_RIGHT:
            obj.manual_force=100


##############################################################################
# Pymunk
def addTrack(screen_width, space, track_pos_y, padding):
    track_body, track_shape = getTrack(
        screen_width,
        padding
    )
    track_body.position = (-padding / 2, track_pos_y)
    space.add(track_shape, track_body)
    return track_body, track_shape

def addCart( 
        screen_width,
        space,
        cart_width, 
        cart_height, 
        cart_mass,
        cart_x,
        track_pos_y
    ):
    cart_body, cart_shape = getCart(
        cart_width,
        cart_height,
        cart_mass
    )
    cart_body.position = (
        cart_x - (cart_width / 2),
        track_pos_y
    )
    space.add(cart_shape, cart_body)
    return cart_body, cart_shape

def addPole(
        screen_width,
        space,
        pole_length,
        pole_mass,
        track_pos_y,
        cart_x,
        cart_height
    ):
    pole_body, pole_shape = getPole(pole_length, pole_mass)
    pole_body.position = (
        cart_x,
        track_pos_y + (cart_height / 2)
    )
    space.add(pole_shape, pole_body)
    return pole_body, pole_shape

def addConstraints(space, cart_shape, track_shape, pole_shape):
    constraints = getCartConstraints(
        cart_shape,
        track_shape,
        pole_shape
    )
    space.add(*constraints)
    return constraints

def getShapeWidthHeight(shape):
    bb = shape.bb
    return ((bb.right - bb.left), (bb.top - bb.bottom))

def getCartConstraints(cart_shape, track_shape, pole_shape):
    cart_width, cart_height = getShapeWidthHeight(cart_shape)
    track_width, _ = getShapeWidthHeight(track_shape)
    track_c_1 = pymunk.GrooveJoint(
        track_shape.body,
        cart_shape.body,
        (0, 0),  # Groove start on track
        (track_width, 0),  # Groove end on track
        # Body local anchor on cart
        (0, 0)
    )
    # Make constraints as 'strong' as possible
    track_c_1.error_bias = 0.0001
    track_c_2 = pymunk.GrooveJoint(
        track_shape.body,
        cart_shape.body,
        (0, 0),  # Groove start on track
        (track_width, 0),  # Groove end on track
        # Body local anchor on cart
        (cart_width, 0)
    )
    track_c_2.error_bias = 0.0001
    cart_pole_c = pymunk.PivotJoint(
        cart_shape.body,
        pole_shape.body,
        # Body local anchor on cart
        (cart_width / 2, cart_height / 2),
        # Body local achor on pole
        (0, 0)
    )
    cart_pole_c.error_bias = 0.0001

    return (track_c_1, track_c_2, cart_pole_c)

def getPole(length, mass, friction=1.0):
    body = pymunk.Body(0, 0)
    shape = pymunk.Segment(
        body,
        (0, 0),
        (0, length),
        5
    )
    shape.sensor = True  # Disable collision
    shape.mass = mass
    shape.friction = friction
    return (body, shape)


def getTrack(track_length, padding):
    track_body = pymunk.Body(body_type=pymunk.Body.STATIC)
    track_shape = pymunk.Segment(
        track_body,
        (0, 0),
        (track_length + padding, 0),
        2
    )
    track_shape.sensor = True  # Disable collision
    return (track_body, track_shape)


def getCart(width, height, mass):
    inertia = pymunk.moment_for_box(
        mass,
        (width, height)
    )
    body = pymunk.Body(mass, inertia)
    shape = getPymunkRect(
        body,
        width,
        height
    )
    return (body, shape)

def getPymunkRect(body, width, height):
    shape = pymunk.Poly(body, [
      (0, 0),
      (width, 0),
      (width, height),
      (0, height)
    ])
    return shape
