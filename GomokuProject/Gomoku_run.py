import gym
import time
import numpy as np
from Gomoku import Gomoku
env=Gomoku.GomokuEnv(10)
env.reset()
done = False
env.action_space.seed(2)
while not done:
    s1 = env.action_space.sample()
    s2 = env.action_space.sample()
    _, _, dones, _=env.step({"agent_1": s1, "agent_2": s2})
    done = dones['__all__']
    env.render()
    time.sleep(3)

input('press any key')
env.viewer.close()
