"""Campaign configuration types for SEAL experiment automation.

Provides typed dataclasses for experiment configuration (ExtensionConfig,
CampaignResult) and the EXAMPLE_WEIGHTS_MAP that resolves paper-original
weight paths for baseline reproduction.

Usage:
    from api.evaluation.campaign_config import (
        ExtensionConfig,
        CampaignResult,
        EXAMPLE_WEIGHTS_MAP,
        resolve_example_weights,
        config_to_dict,
        result_to_dict,
    )

    cfg = ExtensionConfig(
        name="baseline_fedrl_3x3",
        trainer_type="FedRL",
        topology="grid-3x3",
    )
    weights = resolve_example_weights("FedRL", "grid-3x3")
"""
import dataclasses
import json
import os
from dataclasses import dataclass, field
from typing import List, Optional, Any

# ---------------------------------------------------------------------------
# Path helpers
# ---------------------------------------------------------------------------

# BackEnd directory (two levels up from this file: api/evaluation/ -> api/ -> BackEnd/)
_BACKEND_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class ExtensionConfig:
    """Configuration for a single experiment campaign run.

    Captures all parameters needed for the train-then-evaluate pipeline,
    including Phase 4 extension parameters (FedProx mu, cooperative alpha,
    time-of-day curriculum, time encoding).

    Attributes:
        name: Human-readable label, e.g. "fedprox_mu0.1".
        trainer_type: One of "FedRL", "MARL", "SARL".
        topology: Network topology, e.g. "grid-3x3".
        n_episodes: Training episodes (default 50).
        n_eval_runs: Monte Carlo evaluation runs (default 10).
        base_seed: Base random seed for MC runs (default 42).
        ranked: Whether to use ranked observation features.
        horizon: Maximum episode length in simulation steps.
        fedprox_mu: FedProx proximal term coefficient (0.0 = disabled).
        alpha: Cooperative reward shaping coefficient (1.0 = full cooperative).
        time_of_day: Enable time-of-day demand curriculum during training.
        use_time_encoding: Append sin/cos time features to observations.
        weights_path: If set, skip training and evaluate these weights directly.
    """
    name: str
    trainer_type: str
    topology: str
    n_episodes: int = 50
    n_eval_runs: int = 10
    base_seed: int = 42
    ranked: bool = True
    horizon: int = 450
    fedprox_mu: float = 0.0
    alpha: float = 1.0
    time_of_day: bool = False
    use_time_encoding: bool = False
    weights_path: Optional[str] = None


@dataclass
class CampaignResult:
    """Result from a single train-then-evaluate campaign run.

    Attributes:
        config: The ExtensionConfig that produced this result.
        training_rewards: Episode reward data from training. None if
            weights_path was provided (training was skipped).
        evaluation: MCAggregatedResult from Monte Carlo evaluation.
        weights_path: Path to the .pkl file used for evaluation.
        error: Error message if the campaign failed, else None.
        duration_seconds: Wall-clock time for the full campaign run.
    """
    config: ExtensionConfig
    training_rewards: Optional[List[dict]] = None
    evaluation: Optional[Any] = None   # MCAggregatedResult (typed loosely to avoid circular)
    weights_path: Optional[str] = None
    error: Optional[str] = None
    duration_seconds: float = 0.0


# ---------------------------------------------------------------------------
# Example weights map — paper-original weight paths (relative to BackEnd/)
# ---------------------------------------------------------------------------

EXAMPLE_WEIGHTS_MAP = {
    ("FedRL", "grid-3x3"): "example_weights/ICCPS/Final/FedRL/grid-3x3/v3_naive-aggr_ranked.pkl",
    ("FedRL", "grid-5x5"): "example_weights/ICCPS/Final/FedRL/grid-5x5/v3_naive-aggr_ranked.pkl",
    ("MARL", "grid-3x3"): "example_weights/ICCPS/Final/MARL/grid-3x3/v3_ranked.pkl",
    ("MARL", "grid-5x5"): "example_weights/ICCPS/Final/MARL/grid-5x5/v3_ranked.pkl",
    ("SARL", "grid-3x3"): "example_weights/ICCPS/Final/SARL/grid-3x3/v3_ranked.pkl",
    ("SARL", "grid-5x5"): "example_weights/ICCPS/Final/SARL/grid-5x5/v3_ranked.pkl",
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def resolve_example_weights(trainer: str, topology: str) -> Optional[str]:
    """Return the absolute path to paper-original example weights.

    Looks up EXAMPLE_WEIGHTS_MAP for (trainer, topology) and resolves the
    relative path to an absolute path using the BackEnd directory.

    Args:
        trainer: Trainer type string, e.g. "FedRL", "MARL", "SARL".
        topology: Topology name, e.g. "grid-3x3".

    Returns:
        Absolute path to the .pkl file if the entry exists in the map,
        else None (e.g. for grid-7x7 which has no example weights).
    """
    relative = EXAMPLE_WEIGHTS_MAP.get((trainer, topology))
    if relative is None:
        return None
    return os.path.normpath(os.path.join(_BACKEND_DIR, relative))


def config_to_dict(config: ExtensionConfig) -> dict:
    """Serialize ExtensionConfig to a JSON-safe dict.

    Uses dataclasses.asdict() for a flat dict with all fields.

    Args:
        config: ExtensionConfig instance to serialize.

    Returns:
        Dict with all ExtensionConfig fields as plain Python values.
    """
    return dataclasses.asdict(config)


def result_to_dict(result: CampaignResult) -> dict:
    """Serialize CampaignResult to a JSON-safe dict.

    Handles nested MCAggregatedResult (evaluation field) by converting it
    via its own serialization path. Uses json.dumps(default=str) for safety
    on non-serializable values.

    Args:
        result: CampaignResult instance to serialize.

    Returns:
        Dict suitable for json.dumps().
    """
    # Serialize evaluation (MCAggregatedResult) if present
    evaluation_dict = None
    if result.evaluation is not None:
        try:
            from api.evaluation.monte_carlo import campaign_to_dict
            evaluation_dict = campaign_to_dict([result.evaluation])
        except Exception:
            # Fallback: try dataclasses.asdict or str
            try:
                evaluation_dict = dataclasses.asdict(result.evaluation)
            except Exception:
                evaluation_dict = str(result.evaluation)

    raw = {
        "config": config_to_dict(result.config),
        "training_rewards": result.training_rewards,
        "evaluation": evaluation_dict,
        "weights_path": result.weights_path,
        "error": result.error,
        "duration_seconds": result.duration_seconds,
    }

    # Round-trip through json to coerce any non-serializable values to strings
    return json.loads(json.dumps(raw, default=str))
