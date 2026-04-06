"""Federated Distillation trainer — shares knowledge (logits) not weights.

Instead of averaging raw model weights (FedAvg), agents share their action
probability distributions (logits). A consensus distribution is computed
as the mean of all agents' logits, and each agent adds a KL-divergence loss
to match it. This communicates far fewer bytes per round (logits vs full
model weights) while still enabling knowledge transfer across intersections.
"""
import numpy as np
import os
import pickle
import torch

from seal.logging import *
from collections import defaultdict
from seal.sumo.env import SumoEnv
from typing import Any, Dict, NewType, Tuple
from seal.trainer.base import BaseTrainer
from seal.trainer.feddistill_policy import FedDistillPPOTorchPolicy
from seal.trainer.communication.fed_callback import FedRLCommCallback
from seal.trainer.util import *
from time import ctime

Weights = NewType("Weights", Dict[Any, np.array])


class FedDistillTrainer(BaseTrainer):

    def __init__(self, fed_step: int = 1, distill_lambda: float = 0.1,
                 **kwargs) -> None:
        self.distill_lambda = distill_lambda
        super().__init__(
            env=SumoEnv,
            sub_dir="FedDistill",
            **kwargs
        )
        # Override policy_type to use distillation policy
        self.policy_type = FedDistillPPOTorchPolicy
        self.trainer_name = "FedDistill"
        self.fed_step = fed_step
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
        self.communication_callback_cls = FedRLCommCallback
        # Communication cost tracking (bytes)
        self.training_data["comm_bytes_per_round"] = []
        self.training_data["total_comm_bytes"] = []

    def on_make_final_policy(self) -> Weights:
        """Naive average all policies for evaluation."""
        policies = [self.ray_trainer.get_policy(pid)
                    for pid in self.policies if pid != GLOBAL_POLICY_VAR]
        policy_keys = policies[0].get_weights().keys()
        new_weights = {}
        for key in policy_keys:
            weights = np.array([p.get_weights()[key] for p in policies])
            new_weights[key] = sum(
                1 / len(policies) * weights[k]
                for k in range(len(policies))
            )
        return new_weights

    def save_test_policy(self) -> Weights:
        """Save per-agent weights."""
        per_agent_weights = {}
        for pid in self.policies:
            if pid != GLOBAL_POLICY_VAR:
                per_agent_weights[pid] = (
                    self.ray_trainer.get_policy(pid).get_weights()
                )
        ranked_str = "ranked" if self.ranked else "unranked"
        if self.out_prefix is not None:
            ranked_str = f"{self.out_prefix}_{ranked_str}"
        path = os.path.join(self.out_weights_dir, f"{ranked_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump(
                {"__multi_policy__": True, "policies": per_agent_weights}, f
            )
        return self.on_make_final_policy()

    def _is_aggregating_step(self) -> bool:
        if self.fed_step is None:
            return True
        return (self._round + 1) % self.fed_step == 0

    def on_data_recording_step(self) -> None:
        aggregate_this_round = self._is_aggregating_step()

        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("FedDistill")
        self.training_data["fed_round"].append(aggregate_this_round)
        self.training_data["ranked"].append(self.ranked)
        self.training_data["weight_aggr_fn"].append("distill")
        env_runners = self._result.get("env_runners", self._result)
        self.training_data["episode_reward_mean"].append(
            env_runners.get("episode_reward_mean", 0.0))
        self.training_data["episode_len_mean"].append(
            env_runners.get("episode_len_mean", 0.0))

        if aggregate_this_round:
            policy_ids = [pid for pid in self.policies
                          if pid != GLOBAL_POLICY_VAR]

            # Step 1: Collect logits from each agent on a reference observation
            sample_policy = self.ray_trainer.get_policy(policy_ids[0])
            obs_space = sample_policy.observation_space
            ref_obs = np.zeros(obs_space.shape, dtype=np.float32)

            all_logits = []
            for pid in policy_ids:
                policy = self.ray_trainer.get_policy(pid)
                with torch.no_grad():
                    obs_tensor = torch.tensor(ref_obs).unsqueeze(0).float()
                    model_out, _ = policy.model({"obs": obs_tensor})
                    all_logits.append(model_out.cpu().numpy()[0])

            # Step 2: Compute consensus logits (mean of all agents' logits)
            consensus = np.mean(all_logits, axis=0)

            # Step 3: Set consensus on all agents for KL loss
            for pid in policy_ids:
                policy = self.ray_trainer.get_policy(pid)
                if hasattr(policy, "set_consensus_logits"):
                    policy.set_consensus_logits(consensus)
                    policy.set_distill_lambda(self.distill_lambda)

            # Communication cost: each agent sends logits (small) + receives
            # consensus back. Far cheaper than sending full model weights.
            logit_bytes = consensus.nbytes
            n_clients = len(policy_ids)
            round_bytes = logit_bytes * (n_clients + 1)
        else:
            round_bytes = 0

        self.training_data["comm_bytes_per_round"].append(round_bytes)
        prev_total = (self.training_data["total_comm_bytes"][-1]
                      if self.training_data["total_comm_bytes"] else 0)
        self.training_data["total_comm_bytes"].append(prev_total + round_bytes)

    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        dummy_env = self.env(config=self.env_config_fn())
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        return {
            agent_id: (
                self.policy_type,
                obs_space,
                act_space,
                self.policy_config,
            )
            for agent_id in dummy_env._observe()
        }

    def on_logging_step(self) -> None:
        aggregate_this_round = self._is_aggregating_step()
        status = (
            "{}Ep. #{} | ranked={} | fed_round={} | Mean reward: {:6.2f} | "
            "Mean length: {:4.2f} | Saved {} ({})"
        )
        logging.info(status.format(
            "" if self.trainer_name is None else f"[{self.trainer_name}] ",
            self._round + 1,
            self.ranked,
            aggregate_this_round,
            self._get_result_value("episode_reward_mean", 0.0),
            self._get_result_value("episode_len_mean", 0.0),
            self.model_path.split(os.sep)[-1],
            ctime(),
        ))
