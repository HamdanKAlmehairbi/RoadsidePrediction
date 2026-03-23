import math

import numpy as np

from gymnasium import spaces
from seal.sumo.config import *
from seal.sumo.abstract_env import AbstractSumoEnv
from typing import Any, Dict, List, Tuple


class SumoEnv(AbstractSumoEnv):

    def __init__(self, config):
        self.alpha = config.get("alpha", 1.0)
        self.use_time_encoding = config.get("use_time_encoding", False)
        super().__init__(config)

    @property
    def multi_action_space(self) -> spaces.Space:
        return spaces.Dict({
            idx: self.kernel.tls_hub[idx].action_space
            for _, idx in self.kernel.tls_hub.index2id.items()
        })

    @property
    def action_space(self) -> spaces.Space:
        """This is the action space defined for a *single* traffic light. It is
           defined this way to support RlLib more easily.

        Returns:
            Space: Action space for a single traffic light.
        """
        first = self.kernel.tls_hub.index2id[0]
        return self.kernel.tls_hub[first].action_space

    @property
    def observation_space(self) -> spaces.Space:
        """This is the observation space defined for a *single* traffic light. It is
           defined this way to support RlLib more easily.

        Returns:
            Space: Observation space for a single traffic light.
        """
        first = self.kernel.tls_hub.index2id[0]
        base = self.kernel.tls_hub[first].observation_space
        if self.use_time_encoding:
            n = base.shape[0] + 2
            low = np.concatenate([base.low, np.array([-1.0, -1.0], dtype=np.float32)])
            high = np.concatenate([base.high, np.array([1.0, 1.0], dtype=np.float32)])
            return spaces.Box(low=low, high=high, dtype=np.float32)
        return base

    def action_spaces(self, tls_id) -> spaces.Space:
        return self.kernel.tls_hub[tls_id].action_space

    def observation_spaces(self, tls_id) -> spaces.Space:
        return self.kernel.tls_hub[tls_id].observation_space

    def step(self, action_dict: Dict[Any, int]) -> Tuple[Dict, Dict, Dict, Dict, Dict]:
        if action_dict is not None:
            taken_action = self._do_action(action_dict)
        self.kernel.step()
        self.step_counter += 1
        obs = self._observe()
        reward = self._get_all_rewards(obs)
        is_done = self.__get_done()
        # Ray RLLib old API stack uses terminated/truncated dicts with __all__ key
        terminated = {"__all__": is_done and not self.__is_truncated()}
        truncated = {"__all__": is_done and self.__is_truncated()}
        info = {tls.id: {"is_ranked": self.ranked,
                         "veh2tls_comms": tls.get_num_of_controlled_vehicles()}
                for tls in self.kernel.tls_hub}
        return obs, reward, terminated, truncated, info

    def _do_action(self, actions: Dict[Any, int]) -> List[int]:
        """Perform the provided action for each trafficlight.

        Args:
            actions (Dict[Any, int]): The action that each trafficlight will take

        Returns:
            Dict[Any, int]: Returns the action taken -- influenced by which moves are
                legal or not.
        """
        taken_action = actions.copy()
        for tls in self.kernel.tls_hub:
            if self.action_timer.must_change(tls.index) or \
                    (actions[tls.id] == 1 and self.action_timer.can_change(tls.index)):
                tls.next_phase()
                self.action_timer.restart(tls.index)
            else:
                self.action_timer.incr(tls.index)
                taken_action[tls.index] = 0
        return taken_action

    def __get_done(self) -> bool:
        if self.horizon is None:
            return self.kernel.done()
        elif self.step_counter >= self.horizon:
            return True
        else:
            return self.kernel.done()

    def __is_truncated(self) -> bool:
        """True when episode ends due to horizon limit, not natural termination."""
        if self.horizon is not None and self.step_counter >= self.horizon:
            return True
        return False

    def _get_local_reward(self, obs: np.ndarray) -> float:
        """Per-TLS local reward based on occupancy.

        Parameters
        ----------
        obs : np.ndarray
            Numpy array (containing float64 values) representing the observation.

        Returns
        -------
        float
            The local reward for this step.
        """
        return -1 * (obs[LANE_OCCUPANCY] + obs[HALTED_LANE_OCCUPANCY])**2

    def _get_reward(self, obs: np.ndarray) -> float:
        """Backward-compatible wrapper that delegates to _get_local_reward.

        Satisfies the abstract method contract from AbstractSumoEnv.
        """
        return self._get_local_reward(obs)

    def _get_all_rewards(self, obs: dict) -> dict:
        """Compute rewards with optional cooperative blending.

        When alpha=1.0, returns pure local rewards (backward compatible).
        When alpha<1.0, blends local reward with mean neighbor reward.

        Parameters
        ----------
        obs : dict
            Dict mapping TLS id to observation array.

        Returns
        -------
        dict
            Dict mapping TLS id to (possibly blended) reward float.
        """
        local_rewards = {
            tls.id: self._get_local_reward(obs[tls.id])
            for tls in self.kernel.tls_hub
        }
        if self.alpha >= 1.0:
            return local_rewards

        graph = self.kernel.tls_hub.tls_graph
        cooperative_rewards = {}
        for tls in self.kernel.tls_hub:
            neighbors = graph.get(tls.id, [])
            if neighbors:
                neighbor_mean = sum(local_rewards[n] for n in neighbors) / len(neighbors)
            else:
                neighbor_mean = local_rewards[tls.id]
            cooperative_rewards[tls.id] = (
                self.alpha * local_rewards[tls.id]
                + (1.0 - self.alpha) * neighbor_mean
            )
        return cooperative_rewards

    def _observe(self) -> Dict[Any, np.ndarray]:
        """Get the observations across all the trafficlights, indexed by trafficlight id.

        Returns
        -------
        Dict[Any, np.ndarray]
            Observations from each trafficlight.
        """
        obs = {tls.id: tls.get_observation() for tls in self.kernel.tls_hub}
        if self.ranked:
            self._get_ranks(obs, halted=False)
            self._get_ranks(obs, halted=True)
        # Append time encoding AFTER ranking (so rank indices 10-13 are correct)
        if self.use_time_encoding:
            horizon = self.horizon or 3600
            t = self.step_counter
            sin_t = math.sin(2 * math.pi * t / horizon)
            cos_t = math.cos(2 * math.pi * t / horizon)
            for tls_id in obs:
                obs[tls_id] = np.append(obs[tls_id], [sin_t, cos_t]).astype(np.float32)
        # Clean the observation of NaN and (+/-) Inf values.
        for tls in obs:
            for i in range(len(obs[tls])):
                feature = obs[tls][i]
                if feature == np.nan or feature == float('-inf'):
                    obs[tls][i] = 0.0
                elif feature == float('inf'):
                    obs[tls][i] = 1.0
        return obs

    def _get_ranks(self, obs: Dict, halted: bool=False) -> None:
        """Appends global and local ranks to the observations in an inplace fashion.

        Args:
            obs (Dict): Observation provided by a trafficlight.
        """
        if halted:
            pairs = [(tls, tls_state[HALTED_LANE_OCCUPANCY])
                     for tls, tls_state in obs.items()]
            local_index = LOCAL_HALT_RANK
            global_index = GLOBAL_HALT_RANK
        else:
            pairs = [(tls, tls_state[LANE_OCCUPANCY])
                     for tls, tls_state in obs.items()]
            local_index = LOCAL_RANK
            global_index = GLOBAL_RANK
        pairs = sorted(pairs, key=lambda x: x[1], reverse=True)
        graph = self.kernel.tls_hub.tls_graph  # Adjacency list representation.

        # Calculate the GLOBAL ranks for each tls in the road network.
        for global_rank, (tls, _) in enumerate(pairs):
            try:
                obs[tls][global_index] = 1 - (global_rank / (len(graph)-1))
            except ZeroDivisionError:
                obs[tls][global_index] = 1

        # Calculate LOCAL ranks based on global ranks from above.
        for tls in graph:
            local_rank = 0
            for neighbor in graph[tls]:
                if obs[tls][global_index] > obs[neighbor][global_index]:
                    local_rank += 1
            try:
                # We do *not* subtract the denominator by 1 (as we do with global
                # rank) because `len(graph[tls])` does not include `tls` as a
                # node in the sub-network when it should be included. This means that
                # +1 node cancels out the -1 node.
                obs[tls][local_index] = 1 - (local_rank/len(graph[tls]))
            except ZeroDivisionError:
                obs[tls][local_index] = 1
