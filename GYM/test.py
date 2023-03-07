"""evaluate via restoring from policy (preferred, as it's less time-consuming) """

import gymnasium
import tensorflow as tf
from gymnasium import register
from ray.rllib import TFPolicy
from ray.rllib.models import ModelCatalog
from ray.tune import register_env
from PymunkPole import PymunkPole
from models import CartpoleModel
tf.compat.v1.disable_eager_execution()

register(
    id='PymunkPole-v0',
    entry_point='PymunkPole.PymunkPole:PymunkCartPoleEnv',
    max_episode_steps=10000
)

register_env("CP",lambda _: PymunkPole.PymunkCartPoleEnv())
ModelCatalog.register_custom_model("CartpoleModel", CartpoleModel)
my_restored_policy = TFPolicy.from_checkpoint("policy_checkpoint")
CartpoleEnv=gymnasium.make("PymunkPole-v0")
obs, _ = CartpoleEnv.reset()
cur_action = None
total_rev = 0
rew = None
info = None
done = False
truncated=False
while not done:
    actions = my_restored_policy.compute_single_action(obs)
    obs, rew, done, truncated, info = CartpoleEnv.step(actions[0])
    total_rev += rew
    CartpoleEnv.render()
print(total_rev)
CartpoleEnv.close()