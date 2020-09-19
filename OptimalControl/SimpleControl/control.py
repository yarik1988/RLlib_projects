import gym
from gym import spaces
from gym.utils import seeding
import numpy as np
from os import path

class ControlEnv(gym.Env):
    metadata = {
        'render.modes' : ['human', 'rgb_array'],
        'video.frames_per_second' : 30
    }

    def __init__(self):
        self.max_pos = 1.
        self.max_speed = 0.005
        self.max_impulse = 1
        self.viewer = None
        self.gen_bounds = np.array([self.max_pos, self.max_pos, self.max_speed, self.max_speed])
        self.obs_bounds = np.array([self.max_pos,self.max_pos, np.inf, np.inf])
        self.act_bounds=np.array([self.max_impulse, self.max_impulse])
        self.action_space = spaces.Box(low=-self.act_bounds, high=self.act_bounds, dtype=np.float32)
        self.observation_space = spaces.Box(low=-self.obs_bounds, high=self.obs_bounds, dtype=np.float32)
        self.seed()

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def step(self,u):
        x,y,vx,vy = self.state
        u = np.clip(u, -self.act_bounds, self.act_bounds)
        u=u/10000
        vx = vx+u[0]
        vy = vy+u[1]
        x = x+vx
        y = y+vy

        dist=np.sqrt(x*x+y*y)
        if x>self.max_pos:
           x=self.max_pos
           vx=-vx
        if x <-self.max_pos:
           x = -self.max_pos
           vx = -vx
        if y>self.max_pos:
           y=self.max_pos
           vy=-vy
        if y < -self.max_pos:
           y = -self.max_pos
           vy = -vy



        self.state = np.array([x, y, vx, vy])
        return self.state, -dist, False, {}

    def reset(self):
        self.state=self.np_random.uniform(low=-self.gen_bounds, high=self.gen_bounds)
        return self.state

    def render(self, mode='human'):

        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(500,500)
            self.viewer.set_bounds(-self.max_pos, self.max_pos, -self.max_pos,self.max_pos)
            rod = rendering.make_circle(radius=0.02)
            self.transform = rendering.Transform()
            rod.add_attr(self.transform)
            self.viewer.add_geom(rod)

        self.transform.set_translation(self.state[0],self.state[1])
        return self.viewer.render(return_rgb_array = mode=='rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None

