"""CTDE (Centralized Training Decentralized Execution) trainer.

During training, each agent's observation includes the global state (all agents'
observations concatenated), giving the value function a centralized view.
During evaluation, only local observations are used.
"""
import numpy as np
import os
import pickle

from seal.sumo.ctde_env import CTDESumoEnv
from seal.trainer.base import BaseTrainer
from seal.trainer.communication.multi_callback import MultiPolicyCommCallback
from seal.trainer.util import *
from typing import Any, Dict, Tuple


class CTDETrainer(BaseTrainer):

    def __init__(self, **kwargs):
        super().__init__(
            env=CTDESumoEnv,
            sub_dir="CTDE",
            **kwargs
        )
        self.trainer_name = "CTDE"
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
        self.communication_callback_cls = MultiPolicyCommCallback

    def env_config_fn(self) -> Dict[str, Any]:
        config = super().env_config_fn()
        config["ctde_training"] = True  # Enable global state augmentation
        return config

    def on_make_final_policy(self) -> Weights:
        """Naive average for global policy (used in base.py)."""
        policies = [self.ray_trainer.get_policy(pid)
                    for pid in self.policies if pid != GLOBAL_POLICY_VAR]
        policy_keys = policies[0].get_weights().keys()
        new_weights = {}
        for key in policy_keys:
            weights = np.array([p.get_weights()[key] for p in policies])
            new_weights[key] = sum(1 / len(policies) * weights[k]
                                   for k in range(len(policies)))
        return new_weights

    def save_test_policy(self) -> Weights:
        """Save per-agent weights to preserve CTDE specialization."""
        per_agent_weights = {}
        for pid in self.policies:
            if pid != GLOBAL_POLICY_VAR:
                per_agent_weights[pid] = self.ray_trainer.get_policy(pid).get_weights()
        ranked_str = "ranked" if self.ranked else "unranked"
        if self.out_prefix is not None:
            ranked_str = f"{self.out_prefix}_{ranked_str}"
        path = os.path.join(self.out_weights_dir, f"{ranked_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump({
                "__multi_policy__": True,
                "__ctde__": True,
                "policies": per_agent_weights,
            }, f)
        return self.on_make_final_policy()

    def on_data_recording_step(self) -> None:
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("CTDE")
        self.training_data["fed_round"].append(False)
        self.training_data["ranked"].append(self.ranked)
        self.training_data["weight_aggr_fn"].append(None)
        env_runners = self._result.get("env_runners", self._result)
        self.training_data["episode_reward_mean"].append(
            env_runners.get("episode_reward_mean", 0.0))
        self.training_data["episode_len_mean"].append(
            env_runners.get("episode_len_mean", 0.0))
        # No inter-agent communication during training -- global state is in the env
        self.training_data.setdefault("comm_bytes_per_round", []).append(0)
        prev = (self.training_data.get("total_comm_bytes", [0])[-1]
                if self.training_data.get("total_comm_bytes") else 0)
        self.training_data.setdefault("total_comm_bytes", []).append(prev)

    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        dummy_env = self.env(config=self.env_config_fn())
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        return {
            agent_id: (self.policy_type, obs_space, act_space, self.policy_config)
            for agent_id in dummy_env._observe()
        }
