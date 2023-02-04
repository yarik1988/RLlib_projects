import os
import pickle
import time
import gym
import numpy as np
import tensorflow as tf
import ray
from gym import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.algorithms.a3c import a3c
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from PymunkPole import PymunkPole
from models import CartpoleModel

def prepare_obs(my_list:list)->np.array:
    res=np.ones((1,len(my_list),1))
    res[0,:,0]=my_list
    return res

register(
    id='PymunkPole-v0',
    entry_point='PymunkPole.PymunkPole:PymunkCartPoleEnv',
    max_episode_steps=2000
)
ray.init(include_dashboard=False)
register_env("CP",lambda _: PymunkPole.PymunkCartPoleEnv())
ModelCatalog.register_custom_model("CartpoleModel", CartpoleModel)
trainer=Algorithm.from_checkpoint('cartpole_checkpoints/checkpoint_000066')
CartpoleEnv=gym.make("PymunkPole-v0")
obs = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
done = False
while not done:
    cur_action = trainer.compute_single_action(obs,prev_action=cur_action,prev_reward=rew,info=info)
    obs, rew, done, info = CartpoleEnv.step(cur_action)
    total_rev += rew
    CartpoleEnv.render()
    time.sleep(0.01)
print(total_rev)
CartpoleEnv.close()
ray.shutdown()