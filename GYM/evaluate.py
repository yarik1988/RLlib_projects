"""evaluate via restoring from checkpoint """

import ray
import gymnasium
from gymnasium import register
from ray.rllib.algorithms import Algorithm
from ray.rllib.models import ModelCatalog
from ray.tune import register_env

from PymunkPole import PymunkPole
from models import CartpoleModel

register(
    id='PymunkPole-v0',
    entry_point='PymunkPole.PymunkPole:PymunkCartPoleEnv',
    max_episode_steps=2000
)
ray.init(include_dashboard=False)
register_env("CP",lambda _: PymunkPole.PymunkCartPoleEnv())
ModelCatalog.register_custom_model("CartpoleModel", CartpoleModel)
trainer=Algorithm.from_checkpoint('cartpole_checkpoints/checkpoint_000041')
CartpoleEnv=gymnasium.make("PymunkPole-v0")
obs, _ = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
truncated=None
done = False
while not done:
    cur_action = trainer.compute_single_action(obs,prev_action=cur_action,prev_reward=rew,info=info)
    obs, rew, done, truncated, info = CartpoleEnv.step(cur_action)
    total_rev += rew
    CartpoleEnv.render()
print(total_rev)
CartpoleEnv.close()
ray.shutdown()