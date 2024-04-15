import sys
from typing import Dict, Optional, Union
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env.base_env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode import Episode
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from ray.rllib.utils.typing import PolicyID

class LBCallbacks(DefaultCallbacks):

    def on_episode_step(self,
                        *,
                        worker: RolloutWorker,
                        base_env: BaseEnv,
                        episode: Episode,
                        env_index: Union[int, None] = None,
                        **kwargs) -> None:
        episode.custom_metrics["longest_queue_length"] = base_env.envs[0].get_longest_queue()

    def on_episode_end(self,
                    *,
                    worker: RolloutWorker,
                    base_env: BaseEnv,
                    policies: Dict[PolicyID, Policy],
                    episode: EpisodeV2,
                    env_index: Union[int, None] = None,
                    **kwargs) -> None:
        
        # print(dir(base_env.env_states[0]))
        # print(type(base_env.envs[0]))
        # print(base_env.num_envs)
        # print(dir(episode))
        episode.custom_metrics["sim_iterations"] = base_env.envs[0].get_sim_iterations()
        # input("Press Enter to continue...")
    


