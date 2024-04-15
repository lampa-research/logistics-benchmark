import datetime
import os
import random

import numpy as np
from gymnasium.spaces import Box

from marl.marl_environment import MARLEnv

EP_LEN = 50
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

for d in dir(env):
    print(d)

print(env.metadata)

# for i in range(EP_LEN):
#         act_dict = {}
#         # sample random actions for all agents
#         for agent_id in env.get_agent_ids():
#             act_dict[agent_id] = np.array(
#                 [env.action_space_sample([0])[0]], dtype=np.float32)
#         obs, rew, done, ready, info = env.step(act_dict)