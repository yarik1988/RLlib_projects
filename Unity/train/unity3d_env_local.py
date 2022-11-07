import argparse
import os
import tensorflow as tf
import ray
from ray import air, tune
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.env.wrappers.unity3d_env import Unity3DEnv
from ray.rllib.utils.test_utils import check_learning_achieved

parser = argparse.ArgumentParser()
parser.add_argument(
    "--env",
    type=str,
    default="3DBall",    
    help="The name of the Env to run in the Unity3D editor: `3DBall(Hard)?|"
    "Pyramids|GridFoodCollector|SoccerStrikersVsGoalie|Sorter|Tennis|"
    "VisualHallway|Walker` (feel free to add more and PR!)",
)
parser.add_argument(
    "--file-name",
    type=str,
    default=None,
    help="The Unity3d binary (compiled) game, e.g. "
    "'/home/ubuntu/soccer_strikers_vs_goalie_linux.x86_64'. Use `None` for "
    "a currently running Unity3D editor.",
)
parser.add_argument(
    "--from-checkpoint",
    type=str,
    default=None,
    help="Full path to a checkpoint file for restoring a previously saved "
    "Trainer state.",
)
parser.add_argument("--num-workers", type=int, default=0)
parser.add_argument(
    "--as-test",
    action="store_true",
    help="Whether this script should be run as a test: --stop-reward must "
    "be achieved within --stop-timesteps AND --stop-iters.",
)
parser.add_argument(
    "--stop-iters", type=int, default=9999, help="Number of iterations to train."
)
parser.add_argument(
    "--stop-timesteps", type=int, default=10000000, help="Number of timesteps to train."
)
parser.add_argument(
    "--stop-reward",
    type=float,
    default=9999.0,
    help="Reward at which we stop training.",
)
parser.add_argument(
    "--horizon",
    type=int,
    default=3000,
    help="The max. number of `step()`s for any episode (per agent) before "
    "it'll be reset again automatically.",
)


if __name__ == "__main__":
    ray.init()
    args = parser.parse_args()
    tune.register_env(
        "unity3d",
        lambda c: Unity3DEnv(
            file_name=c["file_name"],
            no_graphics=(args.env != "VisualHallway" and c["file_name"] is not None),
            episode_horizon=c["episode_horizon"],
        ),
    )
    # Get policies (different agent types; "behaviors" in MLAgents) and
    # the mappings from individual agents to Policies.
    policies, policy_mapping_fn = Unity3DEnv.get_policy_configs_for_game(args.env)
    config = (
        PPOConfig()
        .environment(
            "unity3d",
            env_config={
                "file_name": args.file_name,
                "episode_horizon": args.horizon,
            },
            disable_env_checking=True,
        )
        .framework(framework="torch")
        # For running in editor, force to use just one Worker (we only have
        # one Unity running)!
        .rollouts(
            num_rollout_workers=args.num_workers if args.file_name else 0,
            no_done_at_end=True,
            rollout_fragment_length=200,
        )
        .training(
            lr=0.0003,
            lambda_=0.95,
            gamma=0.99,
            sgd_minibatch_size=256,
            train_batch_size=4000,
            num_sgd_iter=20,
            clip_param=0.2,
            model={"fcnet_hiddens": [512, 512]},
        )
        .multi_agent(policies=policies, policy_mapping_fn=policy_mapping_fn)
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        .resources(num_gpus=len(tf.config.list_physical_devices('GPU')))
    )
    
    stop = {
        "training_iteration": args.stop_iters,
        "timesteps_total": args.stop_timesteps,
        "episode_reward_mean": args.stop_reward,
    }

    # Run the experiment.
    results = tune.Tuner(
        "PPO",
        param_space=config.to_dict(),
        run_config=air.RunConfig(
            stop=stop,
            verbose=3,
            checkpoint_config=air.CheckpointConfig(
                checkpoint_frequency=1,
                checkpoint_at_end=True,
            ),
        ),
    ).fit()

    # And check the results.
    if args.as_test:
        check_learning_achieved(results, args.stop_reward)
    ray.shutdown()
