"""
Classic cart-pole system.
Pymunk version by Ian Danforth
"""
import time
import math
from typing import Optional

import gym
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from pymunk.vec2d import Vec2d
from gym import spaces, logger
from gym.utils import seeding

from . import cartpole_utils as utils


class PymunkCartPoleEnv(gym.Env):
    metadata = {
        'render.modes': ['human', 'rgb_array'],
        'video.frames_per_second': 50
    }

    def __init__(self):
        # Pygame and display setup
        self.screen = None
        self.draw_options = None
        self.screen_width = 600
        self.screen_height = 400
        self.clock = pygame.time.Clock()
        self.seed()
        self.force_mag = 500.0
        self.manual_force = 0
        self.steps_count = 0
        self._initPymunk()
        # Action Space

        # force_mag here is 50x force_mag in standard cartpole because
        # this force is divided by the frames per second when applied by pymunk

        self.action_space = spaces.Discrete(3)

        # Observation Space
        # Angle at which to fail the episode
        self.theta_threshold_radians = 90 * math.pi / 180

        # Angle limit set to 2 * theta_threshold_radians so failing observation
        # is still within bounds
        high = np.array([
            1,
            np.finfo(np.float32).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float32).max])

        self.observation_space = spaces.Box(-high*2, high*2,dtype=np.double)

        self.steps_beyond_done = None

    def _initPymunk(self):
        # Simulation space
        self.steps_count = 0
        self.space = pymunk.Space()
        self.space.gravity = (0.0, -980.0)
        self.space.iterations = 20  # Double default
        # Track
        track_pos_y = 100
        # Track outside of view area
        padding = 400
        self.track_body, self.track_shape = utils.addTrack(
            self.screen_width,
            self.space,
            track_pos_y,
            padding
        )

        # Cart
        cart_width = 60
        cart_height = 30
        cart_mass = 1.0
        cart_x = (self.screen_width / 2) + self.np_random.uniform(low=-(self.screen_width / 4), high=(self.screen_width / 4))
        self.cart_body, self.cart_shape = utils.addCart(
            self.screen_width,
            self.space,
            cart_width,
            cart_height,
            cart_mass,
            cart_x,
            track_pos_y
        )
        self.cart_body.velocity = Vec2d(self.np_random.uniform(low=-10, high=10), 0.0)
        # Pole
        pole_length = 110
        pole_mass = 0.1
        self.pole_body, self.pole_shape = utils.addPole(
            self.screen_width,
            self.space,
            pole_length,
            pole_mass,
            track_pos_y,
            cart_x,
            cart_height
        )
        self.pole_body.angle = self.np_random.uniform(low=-0.05, high=0.05)
        self.pole_body.angular_velocity = self.np_random.uniform(low=-0.05,high=0.05)

        # Constraints
        self.constraints = utils.addConstraints(
            self.space,
            self.cart_shape,
            self.track_shape,
            self.pole_shape
        )

    def seed(self, seed=int(time.time())):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self, action):
        self.steps_count = self.steps_count+1

        assert self.action_space.contains(action), "%r (%s) invalid" % (action, type(action))
        if action == 0:
            force = self.force_mag
        elif action == 1:
            force = -self.force_mag
        else:
            force = 0
        self.cart_body.apply_force_at_local_point(
            [force,0],
            self.cart_body.center_of_gravity
        )

        self.pole_body.apply_force_at_local_point([self.manual_force,0],self.pole_body.center_of_gravity)

        tau = math.pi * 2
        theta = self.pole_body.angle % tau
        if theta >= math.pi:
            theta = theta - tau
        x = self.cart_body.position[0]
        center_dist = (2*x - self.screen_width) / self.screen_width
        # Out of bounds failure
        done = x < 0.0 or x > self.screen_width
        # Angular failure
        if not done:
            done = theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians

        reward = 1.0-0.1*(abs(center_dist)+abs(theta))

        if done and self.steps_beyond_done is None:
            self.steps_beyond_done = 0
        else:
            if self.steps_beyond_done == 0:
                logger.warn("""
                You are calling 'step()' even though this environment has already returned
                done = True. You should always call 'reset()' once you receive 'done = True'
                Any further steps are undefined behavior.
                """)
                self.steps_beyond_done += 1
                reward = 0.0

        self.space.step(1 / 50.0)

        cart_x_velocity = self.cart_body.velocity[0]
        pole_ang_velocity = self.pole_body.angular_velocity
        obs = (
            center_dist,
            cart_x_velocity/50,
            theta,
            pole_ang_velocity
        )
        return np.array(obs), reward/10.0, done, {}

    def render(self, mode='human'):
        if self.screen == None:
            print('Setting up screen')
            pygame.init()
            self.screen = pygame.display.set_mode(
                (self.screen_width, self.screen_height)
            )

            # Debug draw setup (called in render())
            self.draw_options = pymunk.pygame_util.DrawOptions(self.screen)
            self.draw_options.flags = 3
        pygame.display.set_caption("Frame " + str(self.steps_count))
        utils.handlePygameEvents(self)

        # Redraw all objects
        self.screen.fill((255, 255, 255))
        self.space.debug_draw(self.draw_options)
        pygame.display.flip()
        self.clock.tick(50)

    def reset(self):
        if self.space:
            del self.space
        self._initPymunk()
        center_dist = (2 * self.cart_body.position[0] - self.screen_width) / self.screen_width

        cart_x_velocity = self.cart_body.velocity[0]
        pole_ang_velocity = self.pole_body.angular_velocity
        tau = math.pi * 2
        theta = self.pole_body.angle % tau
        if theta >= math.pi:
            theta = theta - tau
        obs = (
            center_dist,
            cart_x_velocity / 50,
            theta,
            pole_ang_velocity
        )

        return np.array(obs)
