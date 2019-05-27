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

    def __init__(self, board_size,num_in_a_row):
        self.board_size = board_size
        self.num_in_a_row=num_in_a_row
        shape = (self.board_size, self.board_size)  # board_size * board_size

        self.observation_space = spaces.Dict({
            "real_obs": spaces.Box(-np.ones(shape), np.ones(shape), dtype=np.int),
            "action_mask": spaces.Box(np.zeros(self.board_size**2), np.ones(self.board_size**2), dtype=np.int)})
        self.action_space = spaces.Discrete(board_size**2)
        self.viewer = None
        self.new_move = None
        self.nsteps = 0
        self.winning_stride = None
        self.parity = None

    def reset(self):
        self.parity = False
        self.nsteps = 0
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int)
        obs_agent_0 = {"real_obs": self.board, "action_mask":  np.ones(self.board_size**2)}
        obs_agent_1 = {"real_obs": -self.board, "action_mask": np.ones(self.board_size**2)}

        obs = {"agent_0": obs_agent_0, "agent_1": obs_agent_1}
        return obs

    def step(self, action_dict):
        done = False
        cur_agent = "agent_{}".format(int(self.parity))
        other_agent = "agent_{}".format(int(not self.parity))
        self.nsteps = self.nsteps + 1
        action = action_dict[cur_agent]
        self.new_move = (action//self.board_size, action % self.board_size)
        rewards = {"agent_0": 0.01, "agent_1": 0.01}
        if self.board[self.new_move] == 0:
            self.board[self.new_move] = 1-2*self.parity
            if self.check_five(self.new_move):
                rewards[cur_agent] = (0.2 + 10 / self.nsteps)*(1-2*self.parity)
                rewards[other_agent] = -rewards[cur_agent]
                done = True
            if not np.any(self.board == 0):  # Draw. No reward to anyone
                done = True
        else:
            rewards[cur_agent] = -0.1
            self.new_move = None

        self.parity = not self.parity
        mask_act = np.ndarray.flatten(self.board)
        mask_act = (mask_act == 0)
        obs_agent_0 = {"real_obs": self.board, "action_mask":  mask_act}
        obs_agent_1 = {"real_obs": -self.board, "action_mask": mask_act}

        obs = {"agent_0": obs_agent_0, "agent_1": obs_agent_1}
        dones = {"__all__": done}
        infos = {}
        return obs, rewards, dones, infos


    def query_isfive(self, arr, color):
        if len(arr) < self.num_in_a_row:
            return False
        grarr = np.max([sum(1 for i in g) for k, g in groupby(arr) if k == color])
        return np.max(grarr) >= self.num_in_a_row

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
            line.set_color(1, 0, 0)
            line.add_attr(rendering.Transform(translation=(0.5,0.5)))

            self.viewer.add_geom(line)
        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
