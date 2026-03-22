'''
RESOURCES:
    + https://docs.ray.io/en/master/_modules/ray/rllib/evaluation/episode.html
    + https://github.com/ray-project/ray/blob/master/rllib/examples/custom_metrics_and_callbacks.py
'''

from collections import defaultdict
from ray.rllib.algorithms.callbacks import DefaultCallbacks
from ray.rllib.env import BaseEnv
from ray.rllib.evaluation import RolloutWorker
from ray.rllib.evaluation.episode_v2 import EpisodeV2
from ray.rllib.policy import Policy
from seal.trainer.communication import *
from typing import Dict


class BaseCommCallback(DefaultCallbacks):

    def on_episode_start(self, *, worker=None, base_env=None,
                         policies=None, episode, env_index=0, **kwargs) -> None:
        self.comm_cost = defaultdict(int)
        episode.user_data["comm_cost"] = defaultdict(int)

    def on_episode_end(self, *, worker=None, base_env=None,
                       policies=None, episode, env_index=0, **kwargs) -> None:
        for key in self.comm_cost:
            comm_type, policy_id = key
            comm_type = comm_type.replace("_", "-")
            new_key = f"policy_{policy_id}_comm={comm_type}"
            episode.custom_metrics[new_key] = self.comm_cost[key]

            if new_key not in episode.hist_data:
                episode.hist_data[new_key] = []
            episode.hist_data[new_key].append(self.comm_cost[key])

        # Reset the episode's data.
        episode.user_data["comm_cost"] = defaultdict(int)

    def on_train_result(self, *, algorithm=None, result: dict, **kwargs) -> None:
        result["callback_ok"] = True
