from typing import Dict

'''
episode_data = {
    'policy1': {'reward': ..., 'num_vehicles': ...},
    'policy2': {'reward': ..., 'num_vehicles': ...},
    ...
}
'''

def naive_weight_function(episode_data: Dict) -> Dict[str, float]:
    coeffs = {
        policy: 1 / len(episode_data)
        for policy in episode_data
    }
    return coeffs


def neg_reward_weight_function(episode_data: Dict) -> Dict[str, float]:
    total_reward = abs(sum(policy_data["reward"] 
                           for policy_data in episode_data.values()))
    unnormalized_coeffs = {
        policy: total_reward / (policy_data["reward"] - 1)
        for (policy, policy_data) in episode_data.items()
    }
    try:
        coeffs = {
            policy: unnormalized_coeffs[policy] / sum(unnormalized_coeffs.values())
            for policy in episode_data
        }
    except ZeroDivisionError:
        coeffs = naive_weight_function(episode_data)
    return coeffs


def pos_reward_weight_function(episode_data: Dict) -> Dict[str, float]:
    """Shift-and-normalize reward weighting.

    Computes alpha_k = (R_k - R_min) / sum(R_j - R_min), ensuring all
    coefficients are non-negative regardless of reward sign. Falls back
    to naive (uniform) weighting when all rewards are identical.
    """
    rewards = {policy: data["reward"] for policy, data in episode_data.items()}
    r_min = min(rewards.values())
    shifted = {policy: r - r_min for policy, r in rewards.items()}
    total_shifted = sum(shifted.values())
    if total_shifted == 0:
        # All rewards identical — fall back to uniform
        return naive_weight_function(episode_data)
    return {policy: shifted[policy] / total_shifted for policy in episode_data}


def traffic_weight_function(episode_data: Dict) -> Dict[str, float]:
    total_vehicles = sum(policy_data["num_vehicles"] 
                         for policy_data in episode_data.values())
    try:
        coeffs = {
            policy: policy_data["num_vehicles"] / total_vehicles
            for (policy, policy_data) in episode_data.items()
        }
    except ZeroDivisionError:
        coeffs = naive_weight_function(episode_data)
    return coeffs