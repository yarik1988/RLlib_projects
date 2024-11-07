"""evaluate via restoring from checkpoint """

import ray
import gymnasium
from gymnasium import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog
from ray.train import Checkpoint

from custom_models import RegularizedTFModel, RegularizedTorchModel

import argparse

parser = argparse.ArgumentParser()
parser.add_argument(
    "--backend", type=str, default="torch", help="Specify the used ML framework (torch or tf)"
)
args = parser.parse_args()
backend = args.backend
register(id='PymunkPole-v0',entry_point='PymunkPole.PymunkPoleEnv:PymunkCartPoleEnv')
ray.init()
ModelCatalog.register_custom_model("reg_model",
                                   RegularizedTorchModel if backend == "torch" else RegularizedTFModel)
trainer=Algorithm.from_checkpoint(Checkpoint.from_directory('trainer_'+backend))
CartpoleEnv=gymnasium.make("PymunkPole-v0",max_episode_steps=500)
obs, _ = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
truncated=None
done = False
while not done:
    cur_action = trainer.compute_single_action(obs,prev_action=cur_action,prev_reward=rew,info=info)
    obs, rew, terminated, truncated, info = CartpoleEnv.step(cur_action)
    done = terminated or truncated
    total_rev += rew
    CartpoleEnv.render()
print(total_rev)
CartpoleEnv.close()
ray.shutdown()
