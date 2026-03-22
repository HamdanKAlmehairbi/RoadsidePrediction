"""Convenience wrappers for baseline evaluation trials.

Provides clean, named entry points for fixed-time, max-pressure, and RL
evaluation, all backed by run_trial() from runner.py.

Usage:
    from api.evaluation.baselines import run_fixed_time_trial, run_max_pressure_trial, run_rl_trial

    # Run fixed-time controller
    result = run_fixed_time_trial("grid-3x3", seed=42)

    # Run max-pressure controller
    result = run_max_pressure_trial("grid-3x3", seed=42)

    # Run RL trial (auto-discovers weights)
    result = run_rl_trial("FedRL", "grid-3x3", seed=42)
"""
from .runner import run_trial, resolve_weights_path, TrialResult

_RL_TRAINERS = ("FedRL", "MARL", "SARL")


def run_fixed_time_trial(
    topology: str,
    seed: int,
    ranked: bool = True,
    horizon: int = 450,
) -> TrialResult:
    """Run a single fixed-time (timed-phase) evaluation trial.

    SUMO controls signal timing with its default program; no RL inference.

    Args:
        topology: One of "grid-3x3", "grid-5x5", "grid-7x7".
        seed: Seed for deterministic route generation.
        ranked: Whether to use ranked observation features (affects env config).
        horizon: Maximum episode length in steps.

    Returns:
        TrialResult with per-TLS rewards, tripinfo path, step count, and
        vehicle count. comm_costs will be empty for this baseline.
    """
    return run_trial(
        trainer_type="fixed-time",
        topology=topology,
        seed=seed,
        ranked=ranked,
        weights_path=None,
        horizon=horizon,
    )


def run_max_pressure_trial(
    topology: str,
    seed: int,
    ranked: bool = True,
    horizon: int = 450,
) -> TrialResult:
    """Run a single max-pressure evaluation trial.

    Uses queue-based phase selection (max-pressure heuristic) via TraCI.
    Falls back to fixed-time if TraCI is unavailable.

    Args:
        topology: One of "grid-3x3", "grid-5x5", "grid-7x7".
        seed: Seed for deterministic route generation.
        ranked: Whether to use ranked observation features (affects env config).
        horizon: Maximum episode length in steps.

    Returns:
        TrialResult with per-TLS rewards, tripinfo path, step count, and
        vehicle count. comm_costs will be empty for this baseline.
    """
    return run_trial(
        trainer_type="max-pressure",
        topology=topology,
        seed=seed,
        ranked=ranked,
        weights_path=None,
        horizon=horizon,
    )


def run_rl_trial(
    trainer_type: str,
    topology: str,
    seed: int,
    ranked: bool = True,
    weights_path: str = None,
    horizon: int = 450,
) -> TrialResult:
    """Run a single RL evaluation trial (FedRL, MARL, or SARL).

    Auto-discovers weights from trained_weights/ or example_weights/ if
    weights_path is not provided.

    Args:
        trainer_type: One of "FedRL", "MARL", "SARL".
        topology: One of "grid-3x3", "grid-5x5", "grid-7x7".
        seed: Seed for deterministic route generation.
        ranked: Whether to use ranked observation features.
        weights_path: Explicit path to .pkl weights. If None, auto-discovered.
        horizon: Maximum episode length in steps.

    Returns:
        TrialResult with per-TLS rewards, tripinfo path, comm costs, step
        count, and vehicle count.

    Raises:
        ValueError: If trainer_type is not a recognised RL trainer.
        FileNotFoundError: If no weights can be found for this trainer/topology.
    """
    if trainer_type not in _RL_TRAINERS:
        raise ValueError(
            f"run_rl_trial() requires an RL trainer; got '{trainer_type}'. "
            f"Use one of {list(_RL_TRAINERS)}"
        )

    # Auto-discover weights if not explicitly provided
    if weights_path is None:
        weights_path = resolve_weights_path(trainer_type, topology, ranked)

    return run_trial(
        trainer_type=trainer_type,
        topology=topology,
        seed=seed,
        ranked=ranked,
        weights_path=weights_path,
        horizon=horizon,
    )
