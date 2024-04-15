import datetime
import os
import random

import numpy as np
from gymnasium.spaces import Box
import matplotlib.pyplot as plt

from marl.marl_environment import MARLEnv

EP_LEN = 1000
experiment_name = "LogisticsBenchmarkEnvironmentTest"
LOG_DIR = "~/ray_results"
CONFIG_FILE = os.getcwd() + "/maps/01_plant/01_plant.json"

env_config = {
        'filename': CONFIG_FILE,
        "ep_len": EP_LEN,  # number of steps = tasks per episode
        "max_queue_length": 10,
        "verbose": True,
        # "shared_reward": True,
        "log_dir": LOG_DIR,
        "experiment_name": experiment_name,
    }

env = MARLEnv(env_config)

env.reset()

for i in range(EP_LEN):
    act_dict = {}
    # sample random actions for all agents
    for agent_id in env.get_agent_ids():
        act_dict[agent_id] = np.array(
            [env.action_space_sample([0])[0]], dtype=np.float32)
    obs, rew, done, ready, info = env.step(act_dict)

    queue_len, delays = env.get_task_generator_metrics()

    if done['__all__'] == True:
        # Create a figure with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Plot queue_len
        for key, values in queue_len.items():
            ax1.plot(values, label=f'Key {key}')

        ax1.set_xlabel('Index')
        ax1.set_ylabel('Value')
        ax1.set_title('Queue Length')
        ax1.legend()

        # Plot delays as a box plot
        delays_data = [delays[key] for key in delays.keys()]
        ax2.boxplot(delays_data, labels=[f'Key {key}' for key in delays.keys()])

        ax2.set_xlabel('Key')
        ax2.set_ylabel('Value')
        ax2.set_title('Delays')

        # Adjust spacing between subplots
        plt.tight_layout()

        # Display the plot
        plt.show()

        print(delays)

        env.reset()
