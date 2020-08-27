from gym import spaces
import numpy as np
import time
from ray.rllib.env.multi_agent_env import MultiAgentEnv


class PredatorVictim(MultiAgentEnv):
    def __init__(self, params):
        self.params = params
        self.observation_space = spaces.Box(-np.ones(8), np.ones(8), dtype=np.float64)
        self.action_space = spaces.Box(-np.ones(2), np.ones(2), dtype=np.float64)
        self.entities = dict()
        self.n_steps = 0
        self.viewer = None

    @staticmethod
    def create_entity(max_vel, color):
        res = dict()
        res['pos'] = (2*np.random.rand(2)-1)*0.8
        res['vel'] = 2 * np.random.rand(2) - 1
        res['vel'] /= np.linalg.norm(res['vel'])
        res['vel'] *= np.random.rand() * max_vel
        res['max_vel'] = max_vel
        res['color'] = color
        return res

    def reset(self):
        seeder = np.modf(time.time())[0]
        np.random.seed(int(seeder*2**32))
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
        #rewards["predator"] = (1/dist_between+self.params["catch_distance"])*0.01
        #rewards["victim"] = dist_between*0.01
        if dist_between < self.params["catch_distance"]:
            rewards["predator"] = 1/self.n_steps
            rewards["victim"] = -0.1
            done = True
        elif self.n_steps > self.params["max_steps"]:
            done = True
            rewards["predator"] = -0.1
            rewards["victim"] = 0.1
        observation = np.concatenate((self.entities["predator"]["pos"], self.entities["predator"]["vel"],
                                      self.entities["victim"]["pos"], self.entities["victim"]["vel"]))

        obs = {"predator": observation, "victim": observation}
        dones = {"__all__": done}

        return obs, rewards, dones, {}

    def render(self, mode='human'):
        screen_wh = 600
        radius=self.params["catch_distance"]*screen_wh/4
        if self.viewer is None:
            from gym.envs.classic_control import rendering
            self.viewer = rendering.Viewer(screen_wh, screen_wh)
            for key in self.entities:
                self.entities[key]['trans'] = rendering.Transform()
                self.entities[key]['geom'] = rendering.make_circle(radius=radius)
                self.entities[key]['geom'].add_attr(self.entities[key]['trans'])
                self.entities[key]['geom'].set_color(*self.entities[key]['color'])
                self.viewer.add_geom(self.entities[key]['geom'])
        for key in self.entities:
            pos = (self.entities[key]["pos"]+1)*screen_wh/2
            self.entities[key]['trans'].set_translation(pos[0], pos[1])
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
