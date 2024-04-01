import argparse

import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls
from PymunkPole.PymunkPoleEnv import PymunkCartPoleEnv
from custom_models import *

parser = argparse.ArgumentParser()
parser.add_argument(
    "--run", type=str, default="PPO", help="The RLlib-registered algorithm to use."
)
parser.add_argument(
    "--framework",
    choices=["tf", "tf2", "torch"],
    default="torch",
    help="The DL framework specifier.",
)
parser.add_argument(
    "--steps", type=int, default=25, help="How many iterations of training to perform."
)
ray.init()
args = parser.parse_args()
ModelCatalog.register_custom_model("reg_model",
                                   RegularizedTorchModel if args.framework == "torch" else RegularizedTFModel)
config = (
        get_trainable_cls(args.run)
        .get_default_config()
        .environment(PymunkCartPoleEnv,env_config={"max_steps":200})
        .framework(args.framework)
        .training(model={"custom_model": "reg_model","custom_model_config": {}})
        .rollouts(num_rollout_workers=4)
        .resources(num_gpus=0)
    )
trainer = config.build()
print("Press Ctrl+C to interrupt.")
step = 0
while step < args.steps:
    try:
        step = step + 1
        rest = trainer.train()
        cur_reward=rest["episode_reward_mean"]
        print("step {}/{}, avg. reward: {}, avg. episode length: {}".format(step, args.steps, cur_reward, rest["episode_len_mean"]))
    except KeyboardInterrupt:
        print("Interrupted!")
        break


print("Saving and exiting...")
trainer.save("trainer_"+args.framework)
default_policy=trainer.get_policy(policy_id="default_policy")
default_policy.export_checkpoint("policy_"+args.framework)
ray.shutdown()
