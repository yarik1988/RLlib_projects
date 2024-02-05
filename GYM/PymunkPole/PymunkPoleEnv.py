"""
Classic cart-pole system.
Pymunk version by Ian Danforth
"""
import time
import math
from typing import Optional

import gymnasium
import pygame
import pymunk
import pymunk.pygame_util
import numpy as np
from gymnasium.envs.registration import EnvSpec
from pymunk.vec2d import Vec2d
from gymnasium import spaces, logger
from gymnasium.utils import seeding
from ray.rllib.env import EnvContext
import PymunkPole.cartpole_utils as utils

class PymunkCartPoleEnv(gymnasium.Env):
    metadata = {
        'render_modes': ['human', 'rgb_array'],
        'render_fps':60
    }

    def __init__(self, config: EnvContext=None):
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
        self.spec=EnvSpec(id="PymunkCartPoleEnv",max_episode_steps=config["max_steps"]
                if config is not None and "max_steps" in config else None)
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

    def _initPymunk(self):
        # Simulation space
        pymunk.pygame_util.positive_y_is_up = True
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
        reward = 1.0-0.5*abs(center_dist)
        # Out of bounds failure
        terminated = x < 0.0 or x > self.screen_width or theta < -self.theta_threshold_radians or theta > self.theta_threshold_radians
        truncated = False if self.spec.max_episode_steps is None else self.steps_count >= self.spec.max_episode_steps

        self.space.step(1 / 50.0)
        cart_x_velocity = self.cart_body.velocity[0]
        pole_ang_velocity = self.pole_body.angular_velocity
        obs = (
            center_dist,
            cart_x_velocity/50,
            theta,
            pole_ang_velocity
        )
        return np.array(obs,dtype=np.float64), reward/10, terminated, truncated, {}

    def render(self, mode='human'):
        if self.screen is None:
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
        self.clock.tick(60)

    def reset(self,*,
        seed: Optional[int] = None,
        options: Optional[dict] = None):
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

        return np.array(obs,dtype=np.float64), {}
