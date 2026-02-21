"""FedAvg aggregation implementation."""
import torch
from typing import List, Dict
import copy


def fedavg_aggregate(client_updates: List[Dict[str, torch.Tensor]],
                     weights: List[float]) -> Dict[str, torch.Tensor]:
    """Aggregate client model updates using FedAvg (Federated Averaging).

    Reference: McMahan et al., "Communication-Efficient Learning of Deep Networks
    from Decentralized Data", AISTATS 2017.

    Args:
        client_updates: List of state_dicts from clients
        weights: List of weights (typically num_samples per client)

    Returns:
        Aggregated state_dict for global model
    """
    if len(client_updates) == 0:
        raise ValueError("No client updates provided for aggregation")

    if len(client_updates) != len(weights):
        raise ValueError(
            f"Number of updates ({len(client_updates)}) must match "
            f"number of weights ({len(weights)})"
        )

    # Normalize weights to sum to 1
    total_weight = sum(weights)
    if total_weight == 0:
        raise ValueError("Total weight cannot be zero")
    normalized_weights = [w / total_weight for w in weights]

    # Initialize aggregated params with zeros
    aggregated = {}
    for key in client_updates[0].keys():
        aggregated[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)

    # Weighted sum of parameters
    for client_params, weight in zip(client_updates, normalized_weights):
        for key in aggregated.keys():
            aggregated[key] += client_params[key].float() * weight

    return aggregated


def fedavg_aggregate_partial(client_updates: List[Dict[str, torch.Tensor]],
                             weights: List[float],
                             global_params: Dict[str, torch.Tensor],
                             aggregate_keys: List[str]) -> Dict[str, torch.Tensor]:
    """Aggregate only specified layers, keep others from global model.

    Useful for partial model aggregation in heterogeneous FL.

    Args:
        client_updates: List of state_dicts from clients
        weights: Aggregation weights
        global_params: Current global model state_dict
        aggregate_keys: Parameter names to aggregate

    Returns:
        Aggregated state_dict
    """
    # Start with global params
    aggregated = copy.deepcopy(global_params)

    # Normalize weights
    total_weight = sum(weights)
    normalized_weights = [w / total_weight for w in weights]

    # Only aggregate specified keys
    for key in aggregate_keys:
        if key in client_updates[0]:
            aggregated[key] = torch.zeros_like(client_updates[0][key], dtype=torch.float32)
            for client_params, weight in zip(client_updates, normalized_weights):
                aggregated[key] += client_params[key].float() * weight

    return aggregated


def compute_update_delta(original: Dict[str, torch.Tensor],
                        updated: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    """Compute difference between original and updated model params.

    Useful for analyzing model updates in FL.

    Args:
        original: Original state_dict
        updated: Updated state_dict

    Returns:
        Dict with parameter deltas
    """
    delta = {}
    for key in original.keys():
        delta[key] = updated[key] - original[key]
    return delta


def apply_delta(original: Dict[str, torch.Tensor],
               delta: Dict[str, torch.Tensor],
               learning_rate: float = 1.0) -> Dict[str, torch.Tensor]:
    """Apply delta (gradient) to model parameters.

    Args:
        original: Original state_dict
        delta: Parameter deltas
        learning_rate: Scaling factor for delta

    Returns:
        Updated state_dict
    """
    updated = {}
    for key in original.keys():
        updated[key] = original[key] + learning_rate * delta[key]
    return updated


def compute_model_divergence(client_updates: List[Dict[str, torch.Tensor]],
                             global_params: Dict[str, torch.Tensor]) -> Dict[str, float]:
    """Compute divergence metrics between client models and global model.

    Useful for analyzing heterogeneity in FL.

    Args:
        client_updates: List of client state_dicts
        global_params: Global model state_dict

    Returns:
        Dict with divergence statistics
    """
    divergences = []

    for client_params in client_updates:
        total_diff = 0.0
        total_norm = 0.0

        for key in global_params.keys():
            diff = (client_params[key] - global_params[key]).norm(2).item()
            norm = global_params[key].norm(2).item()
            total_diff += diff ** 2
            total_norm += norm ** 2

        divergence = (total_diff ** 0.5) / (total_norm ** 0.5 + 1e-8)
        divergences.append(divergence)

    return {
        'mean_divergence': sum(divergences) / len(divergences),
        'max_divergence': max(divergences),
        'min_divergence': min(divergences),
        'per_client': divergences
    }
