from seal.trainer.communication import *
from seal.trainer.communication.base_callback import BaseCommCallback


class SinglePolicyCommCallback(BaseCommCallback):
    '''
    TRAINER:
        * edge2tls_policy += 0
        * tls2edge_policy += 0
    ENVIRONMENT:
        * edge2tls_action += 1
        * edge2tls_rank   += 1 (if ranked)
        * tls2edge_obs    += 1
        * veh2tls         += 1 (per vehicle)
    '''
    def on_episode_step(self, *, worker=None, base_env=None,
                        episode, env_index=0, **kwargs) -> None:
        agent_ids = set([pair[0] for pair in episode.agent_rewards.keys()])
        for idx in agent_ids:
            info_dict = episode.last_info_for(idx)
            self.comm_cost[EDGE2TLS_POLICY, idx] += 0
            self.comm_cost[TLS2EDGE_POLICY, idx] += 0
            self.comm_cost[EDGE2TLS_ACTION, idx] += 1
            self.comm_cost[EDGE2TLS_RANK, idx] += int(info_dict["is_ranked"])
            self.comm_cost[TLS2EDGE_OBS, idx] += 1
            self.comm_cost[VEH2TLS_COMM, idx] += info_dict["veh2tls_comms"]