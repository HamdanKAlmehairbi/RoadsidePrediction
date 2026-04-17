from copy import deepcopy
from seal.trainer.communication import COMM_TYPES, VEH2TLS_COMM


class DataParser:

    def __init__(self, result_data) -> None:
        self.result_data = deepcopy(result_data)
        # Ray 2.x nests stats under "env_runners"; Ray 1.x has them at top level
        env_runners = self.result_data.get("env_runners", {})
        self._hist_stats = env_runners.get("hist_stats",
                               self.result_data.get("hist_stats", {}))
        self._episodes_this_iter = env_runners.get("episodes_this_iter",
                                       self.result_data.get("episodes_this_iter", 1))

    def episode_reward(self, iteration: int = -1) -> float:
        reward = self._hist_stats["episode_reward"][iteration]
        return reward

    def policy_reward(self, policy_id: str, iteration: int = -1) -> float:
        policy_key = f"policy_{policy_id}_reward"
        if policy_key in self._hist_stats:
            reward = self._hist_stats[policy_key][iteration]
            return reward
        return 0.0

    def episode_comm_cost(self, comm_type: str = None, iteration: int = -1) -> int:
        if comm_type is None:
            return sum(self.episode_comm_cost(c, iteration)
                       for c in COMM_TYPES)
        else:
            assert comm_type in COMM_TYPES
            query = f"comm={comm_type}"
            for key in self._hist_stats:
                if query in key:
                    return self._hist_stats[key][iteration]
            return 0

    def policy_comm_cost(
        self,
        policy_id: str,
        comm_type: str,
        iteration: int = None
    ) -> int:
        policy_key = f"policy_{policy_id}_comm={comm_type}"
        if policy_key not in self._hist_stats:
            return 0
        if iteration is None:
            i = self._episodes_this_iter
            comm_cost = sum(self._hist_stats[policy_key][-i:])
        else:
            comm_cost = self._hist_stats[policy_key][iteration]
        return comm_cost

    def num_vehicles(self, policy_id: str, iteration=None) -> int:
        policy_key = f"policy_{policy_id}_comm={VEH2TLS_COMM}"
        if policy_key not in self._hist_stats:
            return 0
        if iteration is None:
            i = self._episodes_this_iter
            n_vehicles = sum(self._hist_stats[policy_key][-i:])
        else:
            n_vehicles = self._hist_stats[policy_key][iteration]
        return n_vehicles

    @property
    def episode_reward_max(self) -> float:
        env_runners = self.result_data.get("env_runners", {})
        return env_runners.get("episode_reward_max",
                   self.result_data.get("episode_reward_max", 0.0))
