"""CTDE environment wrapper for SumoEnv.

During training, augments each agent's observation with the global state
(all agents' observations concatenated) to provide a centralized training signal.
During evaluation, only local observations are used (decentralized execution).
"""
import numpy as np
from gymnasium import spaces
from seal.sumo.env import SumoEnv


class CTDESumoEnv(SumoEnv):
    """SumoEnv that appends global state to each agent's observation."""

    def __init__(self, config):
        self._ctde_training = config.get("ctde_training", True)
        # _n_agents must be set before super().__init__() because
        # AbstractSumoEnv.__init__() calls reset() -> _augment_obs() which reads it.
        # Use 0 as placeholder; updated after super().__init__() when kernel is ready.
        self._n_agents = 0
        super().__init__(config)
        # Now kernel is initialized — set the real agent count
        self._n_agents = len(self.kernel.tls_hub.ids)

    @property
    def observation_space(self) -> spaces.Space:
        base = super().observation_space
        if not self._ctde_training or self._n_agents == 0:
            return base
        # Augment: local_obs + flattened global state (all agents' base obs)
        local_dim = base.shape[0]
        global_dim = local_dim * self._n_agents
        total_dim = local_dim + global_dim
        low = np.full(total_dim, -np.inf, dtype=np.float32)
        high = np.full(total_dim, np.inf, dtype=np.float32)
        low[:local_dim] = base.low
        high[:local_dim] = base.high
        for i in range(self._n_agents):
            start = local_dim + i * local_dim
            low[start:start + local_dim] = base.low
            high[start:start + local_dim] = base.high
        return spaces.Box(low=low, high=high, dtype=np.float32)

    def _augment_obs(self, obs: dict) -> dict:
        """Append global state to each agent's observation."""
        if not self._ctde_training or self._n_agents == 0:
            return obs
        sorted_ids = sorted(obs.keys())
        global_state = np.concatenate([obs[aid] for aid in sorted_ids])
        augmented = {}
        for aid, local_obs in obs.items():
            augmented[aid] = np.concatenate(
                [local_obs, global_state]
            ).astype(np.float32)
        return augmented

    def reset(self, **kwargs):
        obs, info = super().reset(**kwargs)
        return self._augment_obs(obs), info

    def step(self, action_dict):
        obs, reward, terminated, truncated, info = super().step(action_dict)
        return self._augment_obs(obs), reward, terminated, truncated, info
