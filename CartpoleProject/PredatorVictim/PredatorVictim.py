from gym import spaces
import numpy as np
import time
import gym
from gym.utils import seeding
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PredatorVictim(gym.Env, MultiAgentEnv):

    metadata = {'render.modes': ['human']}

    def __init__(self, **kwargs):

        self.params = kwargs.get("params")
        self.reward_range = (-0.02*np.sqrt(2), 0.02*np.sqrt(2))
        self.observation_space = spaces.Box(-np.ones(8), np.ones(8), dtype=np.float64)
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float64)
        self.entities = dict()
        self.n_steps = 0
        self.seed()
        self.viewer = None
        self.screen_wh = 600

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    def create_entity(self, max_vel, color):
        res = dict()
        res['pos'] = 2*self.np_random.rand(2)-1
        res['vel'] = 2 * self.np_random.rand(2) - 1
        res['vel'] /= np.linalg.norm(res['vel'])
        res['vel'] *= self.np_random.rand() * max_vel
        res['max_vel'] = max_vel
        res['color'] = color
        return res

    def reset(self):
        self.n_steps = 0
        self.entities["predator"] = self.create_entity(self.params['max_predator_vel'], (1, 0, 0))
        self.entities["victim"] = self.create_entity(self.params['max_victim_vel'], (0, 1, 0))

        observation = np.concatenate((self.entities["predator"]["pos"], self.entities["predator"]["vel"],
                                      self.entities["victim"]["pos"], self.entities["victim"]["vel"]))
        obs = {"predator": observation, "victim": observation}
        return obs

    def step(self, action_dict):
        action_dict['predator'] *= self.params['max_predator_acceleration']
        action_dict['victim'] *= self.params['max_victim_acceleration']
        done = False
        self.n_steps += 1
        rewards = {"predator": 0, "victim": 0}
        for key in self.entities:
            self.entities[key]["vel"] += action_dict[key]
            vel_abs = np.linalg.norm(self.entities[key]["vel"])
            if vel_abs > self.entities[key]["max_vel"]:
                self.entities[key]["vel"] /= vel_abs
                self.entities[key]["vel"] *= self.entities[key]["max_vel"]
            self.entities[key]["pos"] += self.entities[key]["vel"]
            for i in range(2):
                if self.entities[key]["pos"][i] > 1 or self.entities[key]["pos"][i] < -1:
                    self.entities[key]["pos"][i] = np.sign(self.entities[key]["pos"][i])
                    self.entities[key]["vel"][i] *= -1

        dist_between = np.linalg.norm(self.entities["predator"]["pos"]-self.entities["victim"]["pos"])

        rewards["victim"] = dist_between*0.01
        rewards["predator"] = - dist_between * 0.01
        if dist_between < self.params["catch_distance"] or self.n_steps > self.params["max_steps"]:
            done = True

        observation = np.concatenate((self.entities["predator"]["pos"], self.entities["predator"]["vel"],
                                      self.entities["victim"]["pos"], self.entities["victim"]["vel"]))

        obs = {"predator": observation, "victim": observation}
        dones = {"__all__": done}

        return obs, rewards, dones, {}

    def render(self, mode='human'):
        radius=self.params["catch_distance"]*self.screen_wh/4
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(self.screen_wh, self.screen_wh)
            for key in self.entities:
                self.entities[key]['trans'] = rendering.Transform()
                self.entities[key]['geom'] = rendering.make_circle(radius=radius)
                self.entities[key]['geom'].add_attr(self.entities[key]['trans'])
                self.entities[key]['geom'].set_color(*self.entities[key]['color'])
                self.viewer.add_geom(self.entities[key]['geom'])
        for key in self.entities:
            pos = (self.entities[key]["pos"]+1)*self.screen_wh/2
            self.entities[key]['trans'].set_translation(pos[0], pos[1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')

    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
