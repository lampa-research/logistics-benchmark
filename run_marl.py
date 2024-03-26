import os
import random

import numpy as np
from ray import air, tune
from ray.rllib.algorithms.algorithm_config import AlgorithmConfig
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from gymnasium.spaces import Box

from marl.marl_environment import MARLEnv

filename = os.getcwd() + "/maps/00_example/00_example.json"


def one_policy_mapping_fn(agent_id, episode, worker):
    ''' Return policy_0. '''
    return "policy_0"


for i in range(1):

    env_config = {
        'filename': filename,
        # "ep_len": 500,  # number of steps = tasks per episode
        "verbose": True,
        # "shared_reward": True,
    }

    config = (
        # AlgorithmConfig(algo_class=PPO)
        PPOConfig()
        .environment(MARLEnv, env_config=env_config, disable_env_checking=True)
        .rollouts(
            num_rollout_workers=1,
            # num_envs_per_worker=1,
            rollout_fragment_length=128,
            # num_consecutive_worker_failures_tolerance=2
        )
        .framework("torch")
        .training(
            gamma=0.95,
            lr=0.005,
            train_batch_size=128,
            model={
                "vf_share_layers": True,
                "fcnet_hiddens": [128, 128]
            },
        )
        .multi_agent(
            policies_to_train=["policy_0"],
            policies={
                "policy_0": (
                    None,
                    Box(
                        low=np.array([0, 0, 0, 0, 0, 0]),
                        high=np.array([100, 100, 100, 100, 100, 100]),
                        dtype=np.int32,),
                    Box(
                        low=0,
                        high=1,
                        shape=(1,),
                        dtype=np.float32),
                    {})
            },
            policy_mapping_fn=one_policy_mapping_fn,)
        .resources(num_gpus=int(os.environ.get("RLLIB_NUM_GPUS", "0")))
        .debugging(log_level="DEBUG")
    )

    tune.Tuner(
        "PPO",
        run_config=air.RunConfig(
            name="LogisticsBenchmarkTest",
            stop={"training_iteration": 100},
            verbose=3,
            # checkpoint_config=air.CheckpointConfig(checkpoint_frequency=50)
        ),
        param_space=config.to_dict(),
    ).fit()
