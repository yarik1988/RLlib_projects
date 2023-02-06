import os
import keyboard
import ray
import ray.rllib.algorithms.a2c as a2c
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune.registry import register_env
from gym.envs.registration import register
import pickle
import gym
from models import CartpoleModel

ready_to_exit = False
def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True

def create_my_env():
    register(
        id='PymunkPole-v0',
        entry_point='PymunkPole.PymunkPole:PymunkCartPoleEnv',
        max_episode_steps=200
    )
    return gym.make("PymunkPole-v0")

env_creator = lambda config: create_my_env()
ray.init(include_dashboard=False)
register_env("CP", env_creator)
ModelCatalog.register_custom_model("CartpoleModel", CartpoleModel)
#trainer=Algorithm.from_checkpoint('cartpole_checkpoints/checkpoint_000103')
trainer = a2c.A2C(env="CP", config={"model": {"custom_model": "CartpoleModel"},'create_env_on_driver':True})
keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    rest = trainer.train()
    cur_reward=rest["episode_reward_mean"]
    print("avg. reward",cur_reward,"avg. episode length",rest["episode_len_mean"])

trainer.save("cartpole_checkpoints")
default_policy=trainer.get_policy(policy_id="default_policy")
default_policy.export_checkpoint("policy_checkpoint")
ray.shutdown()
