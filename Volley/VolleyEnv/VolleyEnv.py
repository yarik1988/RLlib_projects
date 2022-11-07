from enum import IntEnum

from gym import spaces
import numpy as np
import pygame
import pymunk
import pymunk.pygame_util
import gym
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class Collision(IntEnum):
    FLOOR = 1
    PLAYER_1 = 2
    PLAYER_2 = 3
    BALL = 4


class VolleyEnv(gym.Env, MultiAgentEnv):

    def __init__(self, **kwargs):
        self.params = kwargs.get("params")
        self.observation_space = spaces.Box(-np.ones(4, dtype=np.float32), np.ones(4, dtype=np.float32))
        self.action_space = spaces.Discrete(6)
        self.n_steps = 0
        self.seed()
        self.viewer = None
        self.screen_width = 1000
        self.screen_height = 600
        self.fps = 100

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def get_body_pos(self, body):
        return np.array([body.position.x/self.screen_width,body.position.y/self.screen_height,
                         body.velocity.x/body.max_velocity,body.velocity.y/body.max_velocity])


    def reset(self):
        self._space = pymunk.Space()
        self._space.gravity = (0.0, 3000.0)
        static_body = self._space.static_body
        line_width = 10.0
        static_lines = [
            pymunk.Segment(static_body, (0, -10000), (0, self.screen_height), line_width),
            pymunk.Segment(static_body, (0, self.screen_height), (self.screen_width, self.screen_height), line_width),
            pymunk.Segment(static_body, (self.screen_width, self.screen_height), (self.screen_width, -10000), line_width),
            pymunk.Segment(static_body, (self.screen_width / 2, self.screen_height / 2),
                           (self.screen_width / 2, self.screen_height), line_width),
        ]
        for line in static_lines:
            line.elasticity = 1
            line.friction = 0.6
        static_lines[1].collision_type = int(Collision.FLOOR)
        static_lines[1].friction = 1
        static_lines[1].elsaticity = 0
        self._space.add(*static_lines)
        self.ball = self._create_ball((self.screen_width * 0.25, 50))
        self.can_jump = [False, False]
        player1 = self._create_player(self.screen_width * 0.25, self.screen_height * 0.75, Collision.PLAYER_1)
        player2 = self._create_player(self.screen_width * 0.75, self.screen_height * 0.75, Collision.PLAYER_2)
        self.players = [player1, player2]
        self.n_steps = 0
        observation_p0 = np.concatenate((self.get_body_pos(self.ball),self.get_body_pos(player1["top"])))
        observation_p1 = np.concatenate((self.get_body_pos(self.ball), self.get_body_pos(player1["top"])))
        obs = {"player_0": observation_p0, "player_1": observation_p1}
        return obs

    def _create_ball(self, position):
        """
        Create a ball.
        :return:
        """
        mass = 5
        radius = 30
        inertia = pymunk.moment_for_circle(mass, 0, radius, (0, 0))
        body = pymunk.Body(mass, inertia)
        body.position = position
        body.velocity = 0, 0
        body.max_velocity=1000
        shape = pymunk.Circle(body, radius, (0, 0))
        shape.elasticity = 1
        shape.friction = 0
        shape.collision_type = Collision.BALL
        self._space.add(body, shape)

        handler = self._space.add_collision_handler(Collision.BALL, Collision.FLOOR)

        def tch_begin(arbiter, space, data):
            self.reset()
            return True
        handler.begin = tch_begin


        def limit_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, (0, 1000), damping, dt)
            l = body.velocity.length
            if l > body.max_velocity:
                scale = body.max_velocity / l
                body.velocity = body.velocity * scale

        body.velocity_func = limit_velocity
        return body

    def _create_player(self, posx, posy, col):
        mass = 10
        radius_top = 70
        radius_bottom = 50
        body_top = pymunk.Body(mass, np.inf)
        body_top.position = posx, posy
        body_top.max_velocity = 500
        shape = pymunk.Circle(body_top, radius_top, (0, 0))
        shape.elasticity = 1
        shape.friction = 0
        self._space.add(body_top, shape)
        body_bottom = pymunk.Body(mass, np.inf)
        body_bottom.position = posx, posy + (radius_top + radius_bottom) / 1.5
        body_bottom.max_velocity=500
        shape = pymunk.Circle(body_bottom, radius_bottom, (0, 0))
        shape.elasticity = 0
        shape.friction = 0.2
        shape.collision_type = int(col)
        self._space.add(body_bottom, shape)
        j1 = pymunk.PinJoint(body_top, body_bottom, (20, 0), (-20, 0))
        j2 = pymunk.PinJoint(body_top, body_bottom, (-20, 0), (20, 0))
        j1.collide_bodies = False
        j2.collide_bodies = False
        self._space.add(j1, j2)
        body_bottom.can_jump = False
        body_top.can_jump = False
        handler = self._space.add_collision_handler(col, Collision.FLOOR)

        def tch_begin(arbiter, space, data):
            body_bottom.can_jump = body_top.can_jump = True
            return True

        def tch_end(arbiter, space, data):
            body_bottom.can_jump = body_top.can_jump = False
            return True

        handler.begin = tch_begin
        handler.separate = tch_end

        def limit_velocity(body, gravity, damping, dt):
            pymunk.Body.update_velocity(body, gravity, damping, dt)
            if body.velocity[0] >= 0:
                body.velocity = (min(body.max_velocity, body.velocity[0]), body.velocity[1])
            else:
                body.velocity = (max(-body.max_velocity, body.velocity[0]), body.velocity[1])

        body_bottom.velocity_func = limit_velocity
        body_top.velocity_func = limit_velocity
        return {"top": body_top, "bottom": body_bottom}

    def jump(self, ind):
        player = self.players[ind]["bottom"]
        if player.can_jump:
            player.apply_impulse_at_world_point(pymunk.Vec2d(0, -20000), (0, 0))

    def move_right(self, ind):
        player = self.players[ind]["top"]
        player.apply_impulse_at_local_point(pymunk.Vec2d(2000, 0), (0, 0))

    def move_left(self, ind):
        player = self.players[ind]["top"]
        player.apply_impulse_at_local_point(pymunk.Vec2d(-2000, 0), (0, 0))

    def step(self, action_dict):
        done = False
        self._space.step(1 / self.fps)
        self.n_steps += 1
        rewards = {"predator": 0, "victim": 0}
        observation_p0 = np.concatenate((self.get_body_pos(self.ball),self.get_body_pos(self.players[0]["top"])))
        observation_p1 = np.concatenate((self.get_body_pos(self.ball), self.get_body_pos(self.players[1]["top"])))
        obs = {"player_0": observation_p0, "player_1": observation_p1}
        dones = {"__all__": done}
        return obs, rewards, dones, {}

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = pygame.display.set_mode((self.screen_width, self.screen_height))
            self._draw_options = pymunk.pygame_util.DrawOptions(self.viewer)
            self.clock = pygame.time.Clock()
        self.viewer.fill((255, 255, 255))
        self._space.debug_draw(self._draw_options)
        pygame.display.flip()
        try:
            self.clock.tick(self.fps)
        except KeyboardInterrupt:
            return False
        return True

    def close(self):
        del self._space
        pygame.quit()
