import random
from gym import spaces
import numpy as np
import sys
from ray.rllib.env.multi_agent_env import MultiAgentEnv
from itertools import groupby
import pyglet


class GomokuEnv(MultiAgentEnv):
    win_dim_px=500
    def __init__(self, board_size,num_in_a_row):
        self.board_size = board_size
        self.num_in_a_row=num_in_a_row
        self.obs_shape = (self.board_size, self.board_size, 3)  # board_size * board_size
        self.infos = {i: {"result": 0, "nsteps": 0, "wrong_moves": 0} for i in ['agent_0', 'agent_1']}
        self.observation_space = spaces.Dict({
            "real_obs": spaces.Box(np.zeros(self.obs_shape), np.ones(self.obs_shape)),
            "action_mask": spaces.Box(np.zeros(self.board_size**2), np.ones(self.board_size**2), dtype=bool)})
        self.action_space = spaces.Discrete(board_size**2)
        self.viewer = None
        self.new_move = None
        self.nsteps = 0
        self.winning_stride = None
        self.parity = None

    def cook_obs(self, board):
        res = np.zeros((self.board_size, self.board_size, 3))
        res[:, :, 0] = (board == 1)
        res[:, :, 1] = (board == 0)
        res[:, :, 2] = (board == -1)
        return res

    def reset(self):
        self.infos = {i: {"result": 0, "nsteps": 0, "wrong_moves": 0} for i in ['agent_0', 'agent_1']}
        #self.parity = random.choice([False, True])
        self.parity=False
        self.nsteps = 0
        self.board = np.zeros((self.board_size, self.board_size), dtype=np.int)
        obs_agent_0 = {"real_obs": self.cook_obs(self.board), "action_mask":  np.ones(self.board_size**2)}
        obs_agent_1 = {"real_obs": self.cook_obs(-self.board), "action_mask": np.ones(self.board_size**2)}

        obs = {"agent_0": obs_agent_0, "agent_1": obs_agent_1}
        return obs

    def step(self, action_dict):
        done = False
        cur_agent = "agent_{}".format(int(self.parity))
        other_agent = "agent_{}".format(int(not self.parity))
        self.infos[cur_agent]['nsteps'] += 1
        self.nsteps = self.infos[cur_agent]['nsteps']+self.infos[other_agent]['nsteps']
        action = action_dict[cur_agent]
        self.new_move = (action//self.board_size, action % self.board_size)
        rewards = {cur_agent: 0, other_agent: 0}
        if self.board[self.new_move] == 0:
            self.board[self.new_move] = 1-2*self.parity
            if self.check_five(self.new_move):
                rewards[cur_agent] = 1-self.nsteps/(2*self.board_size**2)
                rewards[other_agent] = -1+self.nsteps/(2*self.board_size**2)
                self.infos[cur_agent]["result"] = 1
                done = True
            elif not np.any(self.board == 0) or self.nsteps>=2*self.board_size**2:  # Draw. No reward to anyone
                rewards[cur_agent] = 0
                rewards[other_agent] = 0
                done = True
        else:
            rewards[cur_agent] = -1  # Incorrect move. Penalty
            self.infos[cur_agent]["wrong_moves"] += 1
            self.new_move = None

        if np.any(self.board == 0):
           act_mask=np.ndarray.flatten(self.board == 0)
        else:
           act_mask=np.ones(self.board_size**2)
        self.parity = not self.parity
        obs_agent_0 = {"real_obs": self.cook_obs(self.board), "action_mask":  act_mask}
        obs_agent_1 = {"real_obs": self.cook_obs(-self.board), "action_mask": act_mask}
        obs = {"agent_0": obs_agent_0, "agent_1": obs_agent_1}
        dones = {"__all__": done}

        return obs, rewards, dones, self.infos


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
        from gym.envs.classic_control import rendering
        from .MyViewer import MyViewer
        if self.viewer is None:
            self.viewer = MyViewer(GomokuEnv.win_dim_px,GomokuEnv.win_dim_px)
            self.viewer.window.on_mouse_press=self.mouse_press
            self.viewer.window.on_key_press=self.key_press	
            self.viewer.set_bounds(0, self.board_size, 0, self.board_size)
            self.label = pyglet.text.Label("", font_size=72, x=GomokuEnv.win_dim_px//2, y=GomokuEnv.win_dim_px//2,
                          anchor_x='center', anchor_y='center',color=(0, 153, 153, 255))
            self.viewer.add_label(self.label)
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
            self.label.text="Pink win!" if self.board[self.new_move] == 1 else "Black win!"	
        elif not np.any(self.board == 0):
            self.label.text="Draw!"  	

        return self.viewer.render(return_rgb_array=mode == 'rgb_array')
    
    def make_mouse_move(self):
       self.hmove=None
       while self.hmove is None or self.board[self.hmove] != 0:
           self.render()
       return self.hmove[0]*self.board_size+self.hmove[1]	

    def mouse_press(self, x, y, button, modifiers):
       cell_x=int(self.board_size*x/self.viewer.window.width)
       cell_y=int(self.board_size*y/self.viewer.window.height)	
       self.hmove=(cell_x,cell_y)
    
    def key_press(self,symbol, modifiers):
        if symbol == pyglet.window.key.ESCAPE:
            self.close()
            sys.exit()
            return True
    
    def close(self):
        if self.viewer:
            self.viewer.close()
            self.viewer = None
