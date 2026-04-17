"""Hierarchical Federated RL trainer — two-tier aggregation.

Two-level aggregation that respects spatial locality:
1. Intra-cluster average: agents in the same spatial cluster average weights
2. Inter-cluster average: cluster averages are merged into one global model
3. All agents receive the global model

This tests whether structured aggregation (respecting spatial locality) beats
flat FedAvg.  Communication cost is intra-cluster (cluster_size * model_bytes
per cluster) + inter-cluster (n_clusters * model_bytes), which can be LESS
than FedRL's N * model_bytes when clusters are small.
"""

import numpy as np
import os
import xml.etree.ElementTree as ET

from collections import defaultdict
from seal.logging import *
from seal.sumo.env import SumoEnv
from seal.trainer.base import BaseTrainer
from seal.trainer.communication.multi_callback import MultiPolicyCommCallback
from seal.trainer.util import *
from time import ctime
from typing import Any, Dict, NewType, Tuple

Weights = NewType("Weights", Dict[Any, np.array])


class HierFedTrainer(BaseTrainer):

    def __init__(self, fed_step: int = 1, **kwargs) -> None:
        super().__init__(
            env=SumoEnv,
            sub_dir="HierFed",
            **kwargs,
        )
        self.trainer_name = "HierFed"
        self.fed_step = fed_step
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
        self.communication_callback_cls = MultiPolicyCommCallback

        # Communication cost tracking (bytes)
        self.training_data["comm_bytes_per_round"] = []
        self.training_data["total_comm_bytes"] = []

        # Lazily built neighbor graph and spatial clusters
        self._neighbor_graph = None
        self._clusters = None

    # ------------------------------------------------------------------ #
    # Neighbor graph (same pattern as gossip_agent.py)
    # ------------------------------------------------------------------ #

    def _build_neighbor_graph(self) -> Dict[str, list]:
        """Build adjacency from SUMO net file.

        Two traffic-light junctions are neighbors if a direct edge connects
        them in either direction.
        """
        graph: Dict[str, list] = {}
        tls_ids: set = set()
        tree = ET.parse(self.net_file)
        for j in tree.findall("junction"):
            if j.attrib.get("type") == "traffic_light":
                tls_ids.add(j.attrib["id"])
        for tls_id in tls_ids:
            neighbors: set = set()
            for e in tree.findall("edge"):
                frm = e.attrib.get("from")
                to = e.attrib.get("to")
                if frm == tls_id and to in tls_ids:
                    neighbors.add(to)
                elif to == tls_id and frm in tls_ids:
                    neighbors.add(frm)
            graph[tls_id] = list(neighbors)
        return graph

    # ------------------------------------------------------------------ #
    # Spatial clustering
    # ------------------------------------------------------------------ #

    def _build_clusters(self):
        """Group agents into spatial clusters based on adjacency.

        Uses simple greedy BFS: start from an unassigned agent, add all
        neighbors breadth-first, capping each cluster at roughly half the
        total agents.  Produces 2-4 clusters for typical grid networks.
        """
        graph = self._neighbor_graph
        if graph is None:
            graph = self._build_neighbor_graph()
            self._neighbor_graph = graph

        unassigned = set(graph.keys())
        clusters = []
        while unassigned:
            # Start a new cluster from an arbitrary unassigned agent
            seed = next(iter(unassigned))
            cluster = {seed}
            unassigned.remove(seed)
            # BFS: add all unassigned neighbors up to max_size
            frontier = [seed]
            max_size = max(2, len(graph) // 2)
            while frontier and len(cluster) < max_size:
                node = frontier.pop(0)
                for neighbor in graph.get(node, []):
                    if neighbor in unassigned:
                        cluster.add(neighbor)
                        unassigned.remove(neighbor)
                        frontier.append(neighbor)
            clusters.append(list(cluster))
        return clusters

    # ------------------------------------------------------------------ #
    # Policy setup (per-agent, same as MARL / FedRL / Gossip)
    # ------------------------------------------------------------------ #

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

    # ------------------------------------------------------------------ #
    # Final policy — naive average of all agents (for evaluation)
    # ------------------------------------------------------------------ #

    def on_make_final_policy(self) -> Weights:
        policies = [
            self.ray_trainer.get_policy(pid)
            for pid in self.policies
            if pid != GLOBAL_POLICY_VAR
        ]
        param_keys = policies[0].get_weights().keys()
        new_weights = {}
        n = len(policies)
        for key in param_keys:
            stacked = [np.array(p.get_weights()[key]) for p in policies]
            new_weights[key] = sum(stacked) / n
        return new_weights

    # ------------------------------------------------------------------ #
    # Data recording + hierarchical aggregation
    # ------------------------------------------------------------------ #

    def on_data_recording_step(self) -> None:
        aggregate_this_round = self._is_aggregating_step()

        # --- Record training metrics ---
        env_runners = self._result.get("env_runners", self._result)
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("HierFed")
        self.training_data["fed_round"].append(aggregate_this_round)
        self.training_data["ranked"].append(self.ranked)
        self.training_data["weight_aggr_fn"].append(None)
        self.training_data["episode_reward_mean"].append(
            env_runners.get("episode_reward_mean", 0.0))
        self.training_data["episode_len_mean"].append(
            env_runners.get("episode_len_mean", 0.0))

        # --- Hierarchical aggregation ---
        if aggregate_this_round:
            if self._clusters is None:
                self._clusters = self._build_clusters()

            policy_ids = [pid for pid in self.policies if pid != GLOBAL_POLICY_VAR]

            # Step 1: Intra-cluster averaging
            cluster_averages = []
            for cluster in self._clusters:
                cluster_pids = [pid for pid in cluster if pid in policy_ids]
                if not cluster_pids:
                    continue
                # Naive average within cluster
                param_keys = list(
                    self.ray_trainer.get_policy(cluster_pids[0]).get_weights().keys()
                )
                cluster_avg = {}
                for key in param_keys:
                    weights = [
                        np.array(self.ray_trainer.get_policy(pid).get_weights()[key])
                        for pid in cluster_pids
                    ]
                    cluster_avg[key] = sum(weights) / len(weights)
                cluster_averages.append(cluster_avg)
                # Set intra-cluster average on cluster members
                for pid in cluster_pids:
                    self.ray_trainer.get_policy(pid).set_weights(cluster_avg)

            # Step 2: Inter-cluster global averaging
            if len(cluster_averages) > 1:
                param_keys = list(cluster_averages[0].keys())
                global_avg = {}
                for key in param_keys:
                    global_avg[key] = (
                        sum(ca[key] for ca in cluster_averages) / len(cluster_averages)
                    )
                # Broadcast global average to all agents
                for pid in policy_ids:
                    self.ray_trainer.get_policy(pid).set_weights(global_avg)

        # --- Communication cost tracking (bytes) ---
        if aggregate_this_round:
            sample_policy = next(
                self.ray_trainer.get_policy(pid)
                for pid in self.policies if pid != GLOBAL_POLICY_VAR
            )
            model_bytes = sum(
                np.array(v).nbytes for v in sample_policy.get_weights().values()
            )
            # Intra-cluster: each agent sends to its cluster (sum of cluster sizes)
            # Inter-cluster: one model per cluster sent to coordinator
            n_clusters = len(self._clusters) if self._clusters else 1
            policy_ids = [pid for pid in self.policies if pid != GLOBAL_POLICY_VAR]
            intra_bytes = model_bytes * len(policy_ids)  # each agent uploads once
            inter_bytes = model_bytes * n_clusters        # cluster avgs to global
            broadcast_bytes = model_bytes * len(policy_ids)  # global back to all
            round_bytes = intra_bytes + inter_bytes + broadcast_bytes
        else:
            round_bytes = 0

        self.training_data["comm_bytes_per_round"].append(round_bytes)
        prev_total = (
            self.training_data["total_comm_bytes"][-1]
            if self.training_data["total_comm_bytes"]
            else 0
        )
        self.training_data["total_comm_bytes"].append(prev_total + round_bytes)

    # ------------------------------------------------------------------ #
    # Save per-agent weights with __multi_policy__ format
    # ------------------------------------------------------------------ #

    def save_test_policy(self) -> Weights:
        import pickle

        # Save per-agent weights in multi-policy format
        policy_ids = [pid for pid in self.policies if pid != GLOBAL_POLICY_VAR]
        multi_data = {
            "__multi_policy__": True,
            "policies": {
                pid: self.ray_trainer.get_policy(pid).get_weights()
                for pid in policy_ids
            },
        }
        ranked_str = "ranked" if self.ranked else "unranked"
        if self.out_prefix is not None:
            ranked_str = f"{self.out_prefix}_{ranked_str}"
        with open(os.path.join(self.out_weights_dir, f"{ranked_str}.pkl"), "wb") as f:
            pickle.dump(multi_data, f)

        # Return naive-averaged weights for the global policy
        return self.on_make_final_policy()

    # ------------------------------------------------------------------ #
    # Logging
    # ------------------------------------------------------------------ #

    def on_logging_step(self) -> None:
        agg = self._is_aggregating_step()
        status = (
            "{}Ep. #{} | ranked={} | hierfed_round={} | "
            "Mean reward: {:6.2f} | Mean length: {:4.2f} | Saved {} ({})"
        )
        logging.info(status.format(
            "" if self.trainer_name is None else f"[{self.trainer_name}] ",
            self._round + 1,
            self.ranked,
            agg,
            self._get_result_value("episode_reward_mean", 0.0),
            self._get_result_value("episode_len_mean", 0.0),
            self.model_path.split(os.sep)[-1],
            ctime(),
        ))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_aggregating_step(self) -> bool:
        if self.fed_step is None:
            return True
        return (self._round + 1) % self.fed_step == 0
