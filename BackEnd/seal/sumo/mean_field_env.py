"""Mean Field environment wrapper for SumoEnv.

Augments each agent's observation with the mean action of its topological
neighbors from the previous timestep.  This approximates multi-agent
interaction without weight sharing.
"""
import numpy as np
from gymnasium import spaces
from seal.sumo.env import SumoEnv


class MeanFieldSumoEnv(SumoEnv):
    """SumoEnv with mean neighbor action appended to observations."""

    def __init__(self, config):
        super().__init__(config)
        self._prev_actions = {}  # agent_id -> last action (int)

    @property
    def observation_space(self) -> spaces.Space:
        base = super().observation_space
        # Append 1 dim for mean neighbor action (0.0 to 1.0)
        low = np.concatenate([base.low, np.array([0.0], dtype=np.float32)])
        high = np.concatenate([base.high, np.array([1.0], dtype=np.float32)])
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        self._prev_actions = {aid: 0 for aid in obs}
        # Augment observations with zeros (no previous actions on reset)
        augmented = {}
        for aid, ob in obs.items():
            augmented[aid] = np.concatenate(
                [ob, np.array([0.0], dtype=np.float32)]
            )
        return augmented, info

    def step(self, action_dict):
        obs, reward, terminated, truncated, info = super().step(action_dict)

        graph = self.kernel.tls_hub.tls_graph

        # Augment each agent's observation with mean neighbor action
        augmented = {}
        for aid, ob in obs.items():
            neighbors = graph.get(aid, [])
            if neighbors:
                neighbor_actions = [
                    self._prev_actions.get(n, 0) for n in neighbors
                ]
                mean_action = sum(neighbor_actions) / len(neighbor_actions)
            else:
                mean_action = 0.0
            augmented[aid] = np.concatenate(
                [ob, np.array([mean_action], dtype=np.float32)]
            )

        # Store current actions for next step
        self._prev_actions = dict(action_dict) if action_dict else {}

        return augmented, reward, terminated, truncated, info
