import argparse

import keyboard
import ray
from ray.rllib.models import ModelCatalog
from ray.tune.registry import get_trainable_cls
from PymunkPole.PymunkPoleEnv import PymunkCartPoleEnv
from custom_models import *
ready_to_exit = False
def press_key_exit(_q):
    global ready_to_exit
    ready_to_exit=True

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
keyboard.on_press_key("q", press_key_exit)
while True:
    if ready_to_exit:
        break
    rest = trainer.train()
    cur_reward=rest["episode_reward_mean"]
    print("avg. reward",cur_reward,"avg. episode length",rest["episode_len_mean"])

trainer.save("trainer_"+args.framework)
default_policy=trainer.get_policy(policy_id="default_policy")
default_policy.export_checkpoint("policy_"+args.framework)
ray.shutdown()
