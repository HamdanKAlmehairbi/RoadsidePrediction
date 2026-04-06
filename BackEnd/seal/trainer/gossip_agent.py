"""Gossip RL trainer — decentralized peer-to-peer weight averaging.

Instead of aggregating ALL policies into one global model (star topology,
as in FedRL), each agent averages only with its topological neighbors
(mesh topology).  No central server or coordinator is needed.

Each gossip round, agent i computes:
    w_i = (w_i + sum(w_j for j in neighbors(i))) / (1 + len(neighbors(i)))
"""

import numpy as np
import os
import pickle
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


class GossipPolicyTrainer(BaseTrainer):

    def __init__(self, gossip_step: int = 1, gossip_alpha: float = 0.5,
                 **kwargs) -> None:
        super().__init__(
            env=SumoEnv,
            sub_dir="Gossip",
            **kwargs,
        )
        self.trainer_name = "Gossip"
        self.gossip_step = gossip_step
        self.gossip_alpha = gossip_alpha
        self.idx = self.get_key_count()
        self.incr_key_count()
        self.policy_config = {}
        self.policy_mapping_fn = lambda agent_id, *args, **kwargs: agent_id
        # Gossip uses the same per-step comm callback as MARL (no central
        # policy broadcasts), but we track model-exchange bytes separately.
        self.communication_callback_cls = MultiPolicyCommCallback

        # Communication cost tracking (bytes)
        self.training_data["comm_bytes_per_round"] = []
        self.training_data["total_comm_bytes"] = []

        # Lazily built neighbor graph (from SUMO net file)
        self._neighbor_graph = None

    # ------------------------------------------------------------------ #
    # Neighbor graph
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

    def _get_neighbor_graph(self) -> Dict[str, list]:
        if self._neighbor_graph is None:
            self._neighbor_graph = self._build_neighbor_graph()
        return self._neighbor_graph

    # ------------------------------------------------------------------ #
    # Policy setup (per-agent, same as MARL / FedRL)
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
            stacked = np.array([p.get_weights()[key] for p in policies])
            new_weights[key] = sum(stacked[k] for k in range(n)) / n
        return new_weights

    def save_test_policy(self) -> Weights:
        """Save per-agent weights to preserve gossip-trained specialization."""
        per_agent_weights = {}
        for pid in self.policies:
            if pid != GLOBAL_POLICY_VAR:
                per_agent_weights[pid] = self.ray_trainer.get_policy(pid).get_weights()
        ranked_str = "ranked" if self.ranked else "unranked"
        if self.out_prefix is not None:
            ranked_str = f"{self.out_prefix}_{ranked_str}"
        path = os.path.join(self.out_weights_dir, f"{ranked_str}.pkl")
        with open(path, "wb") as f:
            pickle.dump({"__multi_policy__": True, "policies": per_agent_weights}, f)
        return self.on_make_final_policy()

    # ------------------------------------------------------------------ #
    # Data recording + gossip averaging
    # ------------------------------------------------------------------ #

    def on_data_recording_step(self) -> None:
        gossip_this_round = self._is_gossip_step()

        # --- Record training metrics ---
        env_runners = self._result.get("env_runners", self._result)
        self.training_data["round"].append(self._round)
        self.training_data["trainer"].append("Gossip")
        self.training_data["fed_round"].append(gossip_this_round)
        self.training_data["ranked"].append(self.ranked)
        self.training_data["weight_aggr_fn"].append(None)
        self.training_data["episode_reward_mean"].append(
            env_runners.get("episode_reward_mean", 0.0))
        self.training_data["episode_len_mean"].append(
            env_runners.get("episode_len_mean", 0.0))

        # --- Gossip averaging ---
        if gossip_this_round:
            neighbor_graph = self._get_neighbor_graph()
            policy_ids = [pid for pid in self.policies if pid != GLOBAL_POLICY_VAR]

            # Compute new weights for every agent FIRST to avoid contamination.
            new_weights: Dict[str, dict] = {}
            for pid in policy_ids:
                my_weights = self.ray_trainer.get_policy(pid).get_weights()
                neighbor_ids = neighbor_graph.get(pid, [])
                neighbor_ids = [n for n in neighbor_ids if n in policy_ids]
                if not neighbor_ids:
                    # No neighbors — keep own weights unchanged.
                    continue
                all_weights = [my_weights] + [
                    self.ray_trainer.get_policy(n).get_weights()
                    for n in neighbor_ids
                ]
                averaged = {}
                for key in my_weights.keys():
                    averaged[key] = sum(
                        np.array(w[key]) for w in all_weights
                    ) / len(all_weights)
                new_weights[pid] = averaged

            # Apply all at once.
            for pid, w in new_weights.items():
                self.ray_trainer.get_policy(pid).set_weights(w)

        # --- Communication cost tracking (bytes) ---
        if gossip_this_round:
            sample_policy = next(
                self.ray_trainer.get_policy(pid)
                for pid in self.policies if pid != GLOBAL_POLICY_VAR
            )
            model_bytes = sum(
                np.array(v).nbytes for v in sample_policy.get_weights().values()
            )
            neighbor_graph = self._get_neighbor_graph()
            policy_ids = [pid for pid in self.policies if pid != GLOBAL_POLICY_VAR]
            # Each agent sends its weights to each neighbor (one copy per
            # neighbor link).  Total = sum of degrees * model_bytes.
            total_sends = sum(
                len([n for n in neighbor_graph.get(pid, []) if n in policy_ids])
                for pid in policy_ids
            )
            round_bytes = model_bytes * total_sends
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
    # Logging
    # ------------------------------------------------------------------ #

    def on_logging_step(self) -> None:
        gossip_this_round = self._is_gossip_step()
        status = (
            "{}Ep. #{} | ranked={} | gossip_round={} | "
            "Mean reward: {:6.2f} | Mean length: {:4.2f} | Saved {} ({})"
        )
        logging.info(status.format(
            "" if self.trainer_name is None else f"[{self.trainer_name}] ",
            self._round + 1,
            self.ranked,
            gossip_this_round,
            self._get_result_value("episode_reward_mean", 0.0),
            self._get_result_value("episode_len_mean", 0.0),
            self.model_path.split(os.sep)[-1],
            ctime(),
        ))

    # ------------------------------------------------------------------ #
    # Helpers
    # ------------------------------------------------------------------ #

    def _is_gossip_step(self) -> bool:
        if self.gossip_step is None:
            return True
        return (self._round + 1) % self.gossip_step == 0
