import datetime
import os
import random

import numpy as np
from gymnasium.spaces import Box

from marl.marl_environment import MARLEnv

experiment_name = "LogisticsBenchmarkEnvironmentTest"
LOG_DIR = "~/ray_results"
CONFIG_FILE = os.getcwd() + "/maps/00_example/00_example.json"

env_config = {
        'filename': CONFIG_FILE,
        "ep_len": 500,  # number of steps = tasks per episode
        "max_queue_length": 10,
        "verbose": True,
        # "shared_reward": True,
        "log_dir": LOG_DIR,
        "experiment_name": experiment_name,
    }

env = MARLEnv(env_config)

env.reset()

