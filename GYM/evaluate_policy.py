"""evaluate via restoring from policy (preferred, as it's less time-consuming) """

import gymnasium
from gymnasium import register
from ray.rllib import TFPolicy, TorchPolicy
from ray.rllib.models import ModelCatalog
from custom_models import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "backend", type=str, default="torch", help="Specify the used ML framework (torch or tf)"
)
args = parser.parse_args()
backend = args.backend
register(id='PymunkPole-v0',entry_point='PymunkPole.PymunkPoleEnv:PymunkCartPoleEnv')
ModelCatalog.register_custom_model("reg_model",
                                   RegularizedTorchModel if backend == "torch" else RegularizedTFModel)
my_restored_policy = TorchPolicy.from_checkpoint("policy_"+backend) if backend == "torch" else TFPolicy.from_checkpoint("policy_"+backend)
CartpoleEnv=gymnasium.make("PymunkPole-v0",max_episode_steps=500)
obs, _ = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
done = False
truncated=False
while not done:
    actions = my_restored_policy.compute_single_action(obs)
    obs, rew, terminated, truncated, info = CartpoleEnv.step(actions[0])
    done = terminated or truncated
    total_rev += rew
    CartpoleEnv.render()
print(total_rev)
CartpoleEnv.close()
