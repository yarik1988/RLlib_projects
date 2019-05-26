from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from gym import spaces
from gym.envs.classic_control import rendering
import numpy as np
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from itertools import groupby
import gym


class GomokuEnv(MultiAgentEnv):

    def __init__(self, board_size):
        self.board_size = board_size
        shape = (self.board_size, self.board_size)  # board_size * board_size

        self.observation_space = spaces.Box(-np.ones(shape), np.ones(shape), dtype=np.int)
        self.action_space = spaces.Discrete(board_size**2)
        self.viewer = None
        self.new_move = None
        self.nsteps = 0
        self.winning_stride = None

    def reset(self):
        self.parity = False
        self.nsteps = 0
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int)
        obs = {"agent_1": self.board, "agent_2": -self.board}
        return obs

    def step(self, action_dict):
        done = False
        self.nsteps = self.nsteps + 1
        if not self.parity:
            action = action_dict["agent_1"]
        else:
            action = action_dict["agent_2"]
        self.new_move = (action//self.board_size, action % self.board_size)
        rew = 0
        if self.board[self.new_move] == 0:
            self.board[self.new_move] = 1-2*self.parity
            if self.check_five(self.new_move):
                rew = (20 + 1000 / self.nsteps)*(1-2*self.parity)
                done = True
            if not np.any(self.board == 0):  # Draw. No reward to anyone
                done = True
        else:
            self.new_move = None

        self.parity = not self.parity

        mask_act = np.ndarray.flatten(self.board)
        mask_act = mask_act == 0

        rewards = {"agent_1": rew, "agent_2": -rew}
        obs = {"agent_1": self.board, "agent_2": -self.board}
        dones = {"__all__": done}
        infos = {}
        return obs, rewards, dones, infos


    def query_isfive(self, arr, color):
        if len(arr) < 5:
            return False
        grarr = np.max([sum(1 for i in g) for k, g in groupby(arr) if k == color])
        return np.max(grarr) >= 5

    def check_five(self, cur_put):
        dirs = [[0, 1], [1, 0], [1, 1], [-1, 1]]
        dirQ = np.zeros(4)
        color = self.board[cur_put]
        row = self.board[cur_put[0], :]
        dirQ[0] = self.query_isfive(row, color)
        col = self.board[:, cur_put[1]]
        dirQ[1] = self.query_isfive(col, color)
        shift = cur_put[1] - cur_put[0]
        diag1 = np.diagonal(self.board, shift)
        dirQ[2] = self.query_isfive(diag1, color)
        shift = cur_put[1] + cur_put[0] - self.board_size + 1
        diag2 = np.diagonal(np.rot90(self.board), shift)
        dirQ[3] = self.query_isfive(diag2, color)
        if np.max(dirQ) == 1:
            num = np.argmax(dirQ)
            chkdir = dirs[num]
            ax = bx = cur_put[0]
            ay = by = cur_put[1]
            while 0 <= ax < self.board_size and 0 <= ay < self.board_size and self.board[ax, ay] == color:
                ax = ax + chkdir[0]
                ay = ay + chkdir[1]
            while 0 <= bx < self.board_size and 0 <= by < self.board_size and self.board[bx, by] == color:
                bx = bx - chkdir[0]
                by = by - chkdir[1]
            self.winning_stride=((ax,ay),(bx,by))
        return np.max(dirQ)

    def render(self, mode='human'):
        if self.viewer is None:
            self.viewer = rendering.Viewer(500, 500)
            self.viewer.set_bounds(0, self.board_size, 0, self.board_size)
            for i in range(self.board_size + 1):
                rod = rendering.Line((0, i), (self.board_size, i))
                rod.set_color(0, 0, 0)
                self.viewer.add_geom(rod)
                rod = rendering.Line((i, 0), (i, self.board_size))
                rod.set_color(0, 0, 0)
                self.viewer.add_geom(rod)

        if self.new_move is not None:
            circ = rendering.make_circle(0.5, filled=True)
            col = (1, 0.6, 0.6) if self.board[self.new_move] == 1 else (0, 0, 0)
            circ.set_color(*col)
            circ.add_attr(rendering.Transform(translation=(self.new_move[0] + 0.5, self.new_move[1] + 0.5)))
            self.viewer.add_geom(circ)
        if self.winning_stride is not None:
            line = rendering.Line(*self.winning_stride)
            line.linewidth.stroke=5
            line.set_color(1,0,0)
            line.add_attr(rendering.Transform(translation=(0.5,0.5)))

            self.viewer.add_geom(line)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
