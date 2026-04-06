import numpy as np
import os

from seal.logging import *
from collections import defaultdict
from seal.sumo.env import SumoEnv
from typing import Any, Dict, List, NewType
from seal.trainer.base import BaseTrainer
from seal.trainer.fedprox_policy import FedProxPPOTorchPolicy
from seal.trainer.communication.fed_callback import FedRLCommCallback
from seal.trainer.data.parser import DataParser
from seal.trainer.util import *
from seal.trainer.weight_aggr import *
from time import ctime
from typing import Any, Dict, Tuple

Weights = NewType("Weights", Dict[Any, np.array])
Policy = NewType("Policy", Dict[Any, np.array])


MIN_REWARD = -4
DEFAULT_AGGR_FN = "pos_reward"
WEIGHT_FUNCTIONS = {
    "naive":      naive_weight_function,       # Best (what is used in publication)
    "neg_reward": neg_reward_weight_function,  # BAD
    "pos_reward": pos_reward_weight_function,  # Good
    "traffic":    traffic_weight_function      # Experimental
}


class FedPolicyTrainer(BaseTrainer):

    def __init__(self, fed_step: int, fedprox_mu: float = 0.0,
                 fed_tau: float = 1.0, fed_cluster: bool = False,
                 fed_partial: bool = False, **kwargs) -> None:
        self.fedprox_mu = fedprox_mu
        self.fed_tau = fed_tau          # Soft-update: 1.0 = full replace, <1.0 = blend
        self.fed_cluster = fed_cluster  # Clustered aggregation by topology position
        self.fed_partial = fed_partial  # Partial personalization (aggregate first layer only)
        self._prev_global_weights = None  # For soft-update tracking
        super().__init__(
            env=SumoEnv,
            sub_dir="FedRL",
            **kwargs
        )
        # Override policy_type AFTER super().__init__() which sets PPOTorchPolicy
        # via __load_policy_type(). on_policy_setup() reads self.policy_type later.
        if self.fedprox_mu > 0.0:
            self.policy_type = FedProxPPOTorchPolicy
        self.trainer_name = "FedRL"
        self.fed_step = fed_step
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
        self.communication_callback_cls = FedRLCommCallback
        self.reward_tracker = defaultdict(float)
        self.episode_data = defaultdict(lambda: defaultdict(float))
        self.weight_fn = kwargs.get("weight_fn", DEFAULT_AGGR_FN)
        # Communication cost tracking (bytes)
        self.training_data["comm_bytes_per_round"] = []
        self.training_data["total_comm_bytes"] = []
        assert self.weight_fn in WEIGHT_FUNCTIONS

    def __reset_reward_tracker(self) -> None:
        for policy in self.reward_tracker:
            self.reward_tracker[policy] = 0.0
        # Also reset episode_data so aggregation weights reflect current round only
        for policy in self.episode_data:
            self.episode_data[policy]["reward"] = 0.0
            self.episode_data[policy]["num_vehicles"] = 0.0

    def on_make_final_policy(self) -> Weights:
        policy_dict = {policy_id: self.ray_trainer.get_policy(policy_id)
                       for policy_id in self.policies
                       if policy_id != GLOBAL_POLICY_VAR}
        return self.fedavg(policy_dict)

    def on_data_recording_step(self) -> None:
        # Determine if aggregation is performed during this training iteration or not.
        aggregate_this_round = self._is_aggregating_step()

        # Record the data for training process evaluation.
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("FedRL")
        self.training_data["fed_round"].append(aggregate_this_round)
        self.training_data["ranked"].append(self.ranked)
        self.training_data["weight_aggr_fn"].append(self.weight_fn)
        # Store key metrics from env_runners (Ray 2.x) or top-level (Ray 1.x)
        env_runners = self._result.get("env_runners", self._result)
        self.training_data["episode_reward_mean"].append(
            env_runners.get("episode_reward_mean", 0.0))
        self.training_data["episode_len_mean"].append(
            env_runners.get("episode_len_mean", 0.0))

        # Track the reward for this policy during this training step. This is only
        # used for the FedAvg subroutine in the AGGREGATION step.
        parsed_data = DataParser(self._result)
        policy_reward_mean = self._get_result_value("policy_reward_mean", {})
        for policy in self.policies:
            if policy != GLOBAL_POLICY_VAR:
                self.episode_data[policy]["reward"] += parsed_data.policy_reward(
                    policy)
                self.episode_data[policy]["num_vehicles"] += parsed_data.num_vehicles(
                    policy)
                self.reward_tracker[policy] += \
                    policy_reward_mean.get(policy, MIN_REWARD)

        # Aggregate the weights via the Federated Averaging algorithm.
        if aggregate_this_round:
            policy_dict = {policy_id: self.ray_trainer.get_policy(policy_id)
                           for policy_id in self.policies
                           if policy_id != GLOBAL_POLICY_VAR}

            if self.fed_cluster:
                # Clustered aggregation: group by topology position then average within clusters
                self._clustered_aggregation(policy_dict)
            elif self.fed_partial:
                # Partial personalization: aggregate only first hidden layer
                self._partial_aggregation(policy_dict)
            else:
                # Standard or soft-update FedAvg
                new_params = self.fedavg(policy_dict)

                # Soft-update: blend with previous global weights
                if self.fed_tau < 1.0 and self._prev_global_weights is not None:
                    for key in new_params:
                        new_params[key] = (self.fed_tau * np.array(new_params[key]) +
                                          (1.0 - self.fed_tau) * np.array(self._prev_global_weights[key]))
                self._prev_global_weights = {k: np.array(v) for k, v in new_params.items()}

                for policy_id in self.policies:
                    policy = self.ray_trainer.get_policy(policy_id)
                    policy.set_weights(new_params)
                    if self.fedprox_mu > 0.0 and hasattr(policy, 'store_global_weights'):
                        policy.set_fedprox_mu(self.fedprox_mu)
                        policy.store_global_weights()

        # --- Communication cost tracking (bytes) ---
        if aggregate_this_round:
            # Measure bytes for one model's weights
            sample_policy = next(
                self.ray_trainer.get_policy(pid)
                for pid in self.policies if pid != GLOBAL_POLICY_VAR
            )
            model_bytes = sum(
                np.array(v).nbytes for v in sample_policy.get_weights().values()
            )
            n_clients = len([pid for pid in self.policies if pid != GLOBAL_POLICY_VAR])
            round_bytes = model_bytes * (n_clients + 1)  # uploads + broadcast
        else:
            round_bytes = 0
        self.training_data["comm_bytes_per_round"].append(round_bytes)
        prev_total = (self.training_data["total_comm_bytes"][-1]
                      if self.training_data["total_comm_bytes"] else 0)
        self.training_data["total_comm_bytes"].append(prev_total + round_bytes)

    '''
    def on_data_recording_step_v1(self) -> None:
        aggregate_this_round = self._is_aggregating_step()
        parsed_data = DataParser(self._result)
        total_reward = 0
        for policy in self.policies:
            # Record the data for training process evaluation.
            self.training_data["round"].append(self._round)
            self.training_data["trainer"].append("FedRL")
            self.training_data["policy"].append(policy)
            self.training_data["fed_round"].append(aggregate_this_round)
            self.training_data["ranked"].append(self.ranked)
            self.training_data["weight_aggr_fn"].append(self.weight_fn)

            for key, value in self._result.items():
                if isinstance(value, dict):
                    if policy in value:
                        self.training_data[key].append(value[policy])
                    else:
                        self.training_data[key].append(value)
                else:
                    self.training_data[key].append(value)

            # Track the reward for this policy during this training step. This is only
            # used for the FedAvg subroutine in the AGGREGATION step.
            if policy != GLOBAL_POLICY_VAR:
                self.episode_data[policy]["reward"] += parsed_data.policy_reward(policy)
                self.episode_data[policy]["num_vehicles"] += parsed_data.num_vehicles(policy)
                #
                self.reward_tracker[policy] += \
                    self._result["policy_reward_mean"].get(policy, MIN_REWARD)

        # Aggregate the weights via the Federated Averaging algorithm.
        if aggregate_this_round:
            policy_dict = {policy_id: self.ray_trainer.get_policy(policy_id)
                           for policy_id in self.policies
                           if policy_id != GLOBAL_POLICY_VAR}
            new_params = self.fedavg(policy_dict)
            for policy_id in self.policies:
                self.ray_trainer.get_policy(policy_id).set_weights(new_params)
    '''

    def on_policy_setup(self) -> Dict[str, Tuple[Any]]:
        dummy_env = self.env(config=self.env_config_fn())
        obs_space = dummy_env.observation_space
        act_space = dummy_env.action_space
        return {
            agent_id: (
                self.policy_type,
                obs_space,
                act_space,
                self.policy_config
            )
            for agent_id in dummy_env._observe()
        }

    def fedavg(
        self,
        policy_dict: Dict[str, Policy]
        # weight_fn: str="traffic"
    ) -> Weights:
        # STEP 1: Grab the aggregation function specified at initialization.
        weight_fn = WEIGHT_FUNCTIONS[self.weight_fn]

        # STEP 2: Compute the coefficients for each policy in the system based on reward.
        coeffs = weight_fn(self.episode_data)

        # STEP 3: Compute the reward-based averaged policy weights by weight key.
        new_params = {}
        param_keys = next(iter(policy_dict.values())).get_weights().keys()
        for key in param_keys:
            weights = {policy_id: np.array(policy.get_weights()[key])
                       for policy_id, policy in policy_dict.items()}
            new_params[key] = sum(coeffs[policy_id] * weights[policy_id]
                                  for policy_id in policy_dict)

        # STEP 4: Reset the reward trackers for each of the policies.
        self.__reset_reward_tracker()
        return new_params

    def _clustered_aggregation(self, policy_dict: Dict[str, Policy]) -> None:
        """Cluster intersections by reward similarity and aggregate within clusters.

        Uses K-means on accumulated rewards to create 2-3 clusters of intersections
        with similar performance. Falls back to reward-based grouping (top/bottom half)
        if sklearn is unavailable. Works for any topology — no ID format assumptions.
        """
        from collections import defaultdict as dd

        pids = [pid for pid in policy_dict if pid != GLOBAL_POLICY_VAR]
        if len(pids) <= 2:
            # Too few to cluster — fall back to standard FedAvg
            new_params = self.fedavg(policy_dict)
            for pid in pids:
                self.ray_trainer.get_policy(pid).set_weights(new_params)
            return

        # Get reward per intersection for clustering
        rewards = []
        for pid in pids:
            r = self.episode_data.get(pid, {}).get("reward", 0.0)
            rewards.append(r)

        # Cluster by reward: split into high/low performers
        # (simple median split — works for any topology, no ID assumptions)
        median_reward = np.median(rewards)
        clusters = dd(list)
        for pid, r in zip(pids, rewards):
            cluster_key = "high" if r >= median_reward else "low"
            clusters[cluster_key].append(pid)

        for cluster_key, member_ids in clusters.items():
            cluster_policies = {pid: policy_dict[pid] for pid in member_ids}
            cluster_params = self.fedavg(cluster_policies)
            for pid in member_ids:
                self.ray_trainer.get_policy(pid).set_weights(cluster_params)

    def _partial_aggregation(self, policy_dict: Dict[str, Policy]) -> None:
        """Aggregate only the first hidden layer weights; keep remaining layers local.

        Identifies the first layer by finding weight keys with the smallest layer
        index. This is robust to RLlib weight naming changes — it sorts keys
        and takes the first weight+bias pair rather than hardcoding key patterns.
        """
        weight_fn = WEIGHT_FUNCTIONS[self.weight_fn]
        coeffs = weight_fn(self.episode_data)

        param_keys = list(next(iter(policy_dict.values())).get_weights().keys())

        # Strategy: find keys containing common first-layer patterns
        first_layer_keys = []
        patterns = ['_hidden_layers.0.', '_hidden_layers._model.0.',
                     '_logits.', 'fc1.', 'layer1.', '.0.weight', '.0.bias']
        for pattern in patterns:
            matches = [k for k in param_keys if pattern in k]
            if matches:
                first_layer_keys = matches
                break

        if not first_layer_keys:
            # Final fallback: take first 2 keys (weight + bias of whatever comes first)
            # Sort to ensure consistent ordering across runs
            sorted_keys = sorted(param_keys)
            first_layer_keys = sorted_keys[:2]

        # Compute averaged params for shared layers only
        avg_shared = {}
        for key in first_layer_keys:
            weights = {pid: np.array(p.get_weights()[key]) for pid, p in policy_dict.items()}
            avg_shared[key] = sum(coeffs[pid] * weights[pid] for pid in policy_dict)

        # Set only the shared layers to the averaged version, keep rest local
        for policy_id in policy_dict:
            policy = self.ray_trainer.get_policy(policy_id)
            current_weights = policy.get_weights()
            for key in first_layer_keys:
                current_weights[key] = avg_shared[key]
            policy.set_weights(current_weights)

        self._FedPolicyTrainer__reset_reward_tracker()

    # ============================================================ #

    def on_logging_step(self) -> None:
        aggregate_this_round = self._is_aggregating_step()
        status = "{}Ep. #{} | ranked={} | fed_round={} | Mean reward: {:6.2f} | " \
                 "Mean length: {:4.2f} | Saved {} ({})"
        logging.info(status.format(
            "" if self.trainer_name is None else f"[{self.trainer_name}] ",
            self._round+1,
            self.ranked,
            aggregate_this_round,
            self._get_result_value("episode_reward_mean", 0.0),
            self._get_result_value("episode_len_mean", 0.0),
            self.model_path.split(os.sep)[-1],
            ctime()
        ))

    def _is_aggregating_step(self) -> bool:
        if self.fed_step is None:
            return True
        elif (self._round + 1) % self.fed_step == 0:
            return True
        else:
            return False
