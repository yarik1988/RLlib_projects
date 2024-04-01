import os
import argparse
import logging
# import tensorflow as tf
# import torch
import ray
from gym.spaces import Box, MultiDiscrete
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, ppo
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.policy.policy import PolicySpec
from fix_2024 import fix_graph
parser = argparse.ArgumentParser()
parser.add_argument("--backend", dest="backend", type=str, default="torch")
parser.add_argument("--num-workers", dest="num_workers", type=int, default=8)
parser.add_argument("--max-iterations", dest="max_iterations", type=int, default=200000)
parser.add_argument("--unity-exe-name", dest="unity_exe_name", type=str, default="..\\..\\build\\2DTest\\2Dtest.exe")
args = parser.parse_args()
out_folder = os.path.join(os.getcwd(),"ray_"+args.backend)
if args.backend == "torch":
    import torch
    num_gpus = torch.cuda.device_count()
else:
    import tensorflow as tf
    num_gpus = len(tf.config.list_physical_devices('GPU'))

ray.init(logging_level=logging.FATAL, log_to_driver=False)
tune.register_env(
    "unity3d",
    lambda c: Unity3DEnv(
        file_name=args.unity_exe_name,
        no_graphics=True,
        episode_horizon=512,
    ),
)
game_name = "Valley"
policies = {
    game_name: PolicySpec(observation_space=Box(float("-inf"), float("inf"), (4,)), action_space=MultiDiscrete([3])),
}


def policy_mapping_fn(agent_id, episode, worker, **kwargs):
    return game_name


config = (
    PPOConfig()
    .environment(
        "unity3d",
        env_config={"episode_horizon": 512},
        disable_env_checking=True,
    )
    .rollouts(
        num_rollout_workers=args.num_workers,
        rollout_fragment_length=128,
    )
    .framework(framework=args.backend)
    .training(
        lr_schedule= [
            [0, 0.002],
            [100000, 0.0003],
        ],
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=32,
        num_sgd_iter=4,
        clip_param=0.2,
        kl_coeff=0.5,
        kl_target=0.01,
        entropy_coeff=0.001,
        model={"fcnet_hiddens": [64, 64]},
    )
    .debugging(log_level="ERROR")
    .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
    .resources(num_gpus=num_gpus)
)


# Run the experiment.
tuner = tune.Tuner(
    "PPO",
    param_space=config.to_dict(),
    run_config=air.RunConfig(
        stop={"timesteps_total": args.max_iterations},
        verbose=3,
        local_dir=out_folder,
        checkpoint_config=air.CheckpointConfig(
            checkpoint_frequency=1,
            num_to_keep=5,
            checkpoint_at_end=True,
        ),
    ),
)
results = tuner.fit()
print("Got results")
config.num_rollout_workers = 0
agent = ppo.PPO(config=config, env="unity3d")
agent.restore(results.get_best_result().checkpoint)
agent.get_policy("Valley").export_model(out_folder, onnx=9)
print("Model saved successfully!")
fix_graph(os.path.join(out_folder,"model.onnx"))
print("Fixed model")
ray.shutdown()
