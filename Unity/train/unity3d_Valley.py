import argparse
import logging
import tensorflow as tf
import torch
import ray
from gym.spaces import Box, MultiDiscrete
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig, ppo
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.policy.policy import PolicySpec

parser = argparse.ArgumentParser()
parser.add_argument("--backend", type=str, default="tf")
parser.add_argument("--num_workers", type=int, default=0)
parser.add_argument("--max_iterations", type=int, default=50000)
args = parser.parse_args()
out_folder = "ray_"+args.backend
if args.backend == "torch":
    num_gpus = torch.cuda.device_count()
else:
    num_gpus = len(tf.config.list_physical_devices('GPU'))

ray.init(logging_level=logging.FATAL, log_to_driver=False)
tune.register_env(
    "unity3d",
    lambda c: Unity3DEnv(
        file_name="..\\build\\2DTest\\2Dtest.exe",
        no_graphics=True,
        episode_horizon=500,
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
        env_config={"episode_horizon": 500,'mask': True},
        disable_env_checking=True,
    )
    .rollouts(
        num_rollout_workers=args.num_workers,
        no_done_at_end=True,
        rollout_fragment_length=100,
    )
    .framework(framework=args.backend)
    .training(
        lr=0.0003,
        lambda_=0.95,
        gamma=0.99,
        sgd_minibatch_size=64,
        train_batch_size=128,
        num_sgd_iter=4,
        clip_param=0.2,
        model={"fcnet_hiddens": [20, 20]},
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
config.num_rollout_workers = 0
agent = ppo.PPO(config=config, env="unity3d")
agent.restore(results.get_best_result().checkpoint)
agent.get_policy("Valley").export_model(out_folder, onnx=9)
print("Model saved successfully!")
ray.shutdown()
