"""Cross-topology transfer testing for SEAL evaluation.

Builds a train_topology x test_topology matrix by running each
(train_topo, test_topo) pair and computing metrics.  The transfer gap
metric quantifies how much performance degrades when weights trained on
one topology are evaluated on a different topology.

Usage:
    from api.evaluation.transfer import (
        build_transfer_matrix,
        compute_transfer_gap,
        TransferResult,
        transfer_matrix_to_dict,
    )

    results = build_transfer_matrix("FedRL", ranked=True, seed=42, horizon=450)
    gap = compute_transfer_gap(results)
    serializable = transfer_matrix_to_dict(results)
"""
import logging
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .runner import run_trial, resolve_weights_path, TOPOLOGIES
from .metrics import TrialMetrics, compute_trial_metrics, metrics_to_dict

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data class
# ---------------------------------------------------------------------------

@dataclass
class TransferResult:
    """Result of a single (train_topology, test_topology) evaluation.

    Attributes:
        train_topology: Topology the weights were trained on.
        test_topology: Topology the policy was evaluated on.
        trainer: Trainer type (e.g. "FedRL", "MARL", "SARL").
        metrics: Full TrialMetrics for the test episode.
        is_transfer: True when train_topology != test_topology (off-diagonal).
    """
    train_topology: str
    test_topology: str
    trainer: str
    metrics: TrialMetrics = field(default_factory=TrialMetrics)
    is_transfer: bool = False


# ---------------------------------------------------------------------------
# Matrix builder
# ---------------------------------------------------------------------------

def build_transfer_matrix(
    trainer_type: str,
    ranked: bool = True,
    seed: int = 42,
    horizon: int = 450,
) -> List[TransferResult]:
    """Run all (train_topo x test_topo) pairs and collect TransferResult objects.

    For each pair:
    1. Resolve weights trained on train_topo (skip pair with warning if missing).
    2. Run a single evaluation episode on test_topo using those weights.
    3. Compute TrialMetrics from the trial.

    Args:
        trainer_type: One of "FedRL", "MARL", "SARL", "fixed-time", "max-pressure".
        ranked: Whether to use ranked observation features.
        seed: Deterministic seed for route generation.
        horizon: Maximum episode length in steps.

    Returns:
        List of TransferResult — up to len(TOPOLOGIES)^2 entries.
        Pairs where weights are unavailable are skipped.
    """
    results: List[TransferResult] = []

    for train_topo in TOPOLOGIES:
        # Resolve weights once per train_topo (same weights used for all test_topos)
        try:
            weights_path: Optional[str] = resolve_weights_path(trainer_type, train_topo, ranked)
        except FileNotFoundError as exc:
            logger.warning(
                "Skipping train_topo=%s for trainer=%s: %s",
                train_topo, trainer_type, exc,
            )
            continue

        for test_topo in TOPOLOGIES:
            logger.info(
                "Transfer matrix: trainer=%s train=%s test=%s",
                trainer_type, train_topo, test_topo,
            )
            try:
                trial_result = run_trial(
                    trainer_type=trainer_type,
                    topology=test_topo,
                    seed=seed,
                    ranked=ranked,
                    weights_path=weights_path,
                    horizon=horizon,
                )
                trial_metrics = compute_trial_metrics(trial_result)
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Trial failed (trainer=%s, train=%s, test=%s): %s",
                    trainer_type, train_topo, test_topo, exc,
                )
                continue

            results.append(
                TransferResult(
                    train_topology=train_topo,
                    test_topology=test_topo,
                    trainer=trainer_type,
                    metrics=trial_metrics,
                    is_transfer=(train_topo != test_topo),
                )
            )

    return results


# ---------------------------------------------------------------------------
# Gap metric
# ---------------------------------------------------------------------------

def compute_transfer_gap(results: List[TransferResult]) -> List[dict]:
    """Compute transfer gap metrics grouped by trainer.

    The transfer gap measures how much average reward degrades when
    weights are evaluated on a topology they were not trained on.

    - home_performance: mean reward on diagonal (train == test).
    - transfer_performance: mean reward off-diagonal (train != test).
    - gap: home_performance - transfer_performance  (positive = degradation).
    - gap_pct: gap / |home_performance| * 100.

    Args:
        results: List of TransferResult (typically from build_transfer_matrix).

    Returns:
        List of dicts, one per unique trainer in results:
        {
            "trainer": str,
            "home_reward": float,
            "transfer_reward": float,
            "gap": float,
            "gap_pct": float,
        }
    """
    # Group by trainer
    by_trainer: Dict[str, List[TransferResult]] = {}
    for r in results:
        by_trainer.setdefault(r.trainer, []).append(r)

    gap_list: List[dict] = []

    for trainer, trainer_results in by_trainer.items():
        home_rewards = [r.metrics.mean_reward for r in trainer_results if not r.is_transfer]
        transfer_rewards = [r.metrics.mean_reward for r in trainer_results if r.is_transfer]

        home_perf = (sum(home_rewards) / len(home_rewards)) if home_rewards else 0.0
        transfer_perf = (
            sum(transfer_rewards) / len(transfer_rewards)
        ) if transfer_rewards else 0.0

        gap = home_perf - transfer_perf
        gap_pct = (gap / abs(home_perf) * 100.0) if home_perf != 0.0 else 0.0

        gap_list.append({
            "trainer": trainer,
            "home_reward": round(home_perf, 4),
            "transfer_reward": round(transfer_perf, 4),
            "gap": round(gap, 4),
            "gap_pct": round(gap_pct, 4),
        })

    return gap_list


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------

def transfer_matrix_to_dict(results: List[TransferResult]) -> dict:
    """Convert a list of TransferResult to a JSON-serializable nested structure.

    Output shape:
    {
        "trainer": "<trainer_type>",
        "matrix": {
            "<train_topo>": {
                "<test_topo>": <metrics_dict>,
                ...
            },
            ...
        },
        "transfer_gap": [{"trainer": ..., "home_reward": ..., ...}]
    }

    Args:
        results: List of TransferResult from build_transfer_matrix().

    Returns:
        Nested dict suitable for json.dumps() or API responses.
    """
    if not results:
        return {"trainer": "", "matrix": {}, "transfer_gap": []}

    trainer_type = results[0].trainer

    matrix: Dict[str, Dict[str, dict]] = {}
    for r in results:
        matrix.setdefault(r.train_topology, {})[r.test_topology] = metrics_to_dict(r.metrics)

    return {
        "trainer": trainer_type,
        "matrix": matrix,
        "transfer_gap": compute_transfer_gap(results),
    }
