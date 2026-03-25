"""Monte Carlo orchestration for SEAL evaluation.

Provides run_monte_carlo() to execute N trials per configuration with
different random seeds, aggregate results with mean/std/CI statistics,
and support full 5x3x10 campaign runs.

Usage:
    from api.evaluation.monte_carlo import (
        run_monte_carlo,
        run_full_campaign,
        MCConfig,
        MCAggregatedResult,
        campaign_to_dict,
    )

    config = MCConfig(trainer="FedRL", topology="grid-3x3", n_runs=10)
    result = run_monte_carlo(config)
    print(result.aggregated["avg_waiting_time"]["mean"])

    # Full 5 trainers x 3 topologies x 10 runs campaign
    results = run_full_campaign(n_runs=10)
    serializable = campaign_to_dict(results)
"""
import logging
import math
import statistics
from dataclasses import dataclass, field
from typing import Callable, List, Optional

from .runner import run_trial, resolve_weights_path, TRAINERS, TOPOLOGIES
from .metrics import compute_trial_metrics, metrics_to_dict, TrialMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class MCConfig:
    """Configuration for a single Monte Carlo evaluation run.

    Attributes:
        trainer: One of "FedRL", "MARL", "SARL", "fixed-time", "max-pressure".
        topology: Topology name, e.g. "grid-3x3".
        n_runs: Number of Monte Carlo repetitions (default 10, matches SEAL paper).
        base_seed: First random seed; subsequent seeds are base_seed + i.
        ranked: Whether to use ranked observation features.
        horizon: Maximum episode length in simulation steps.
        weights_path: Explicit .pkl path for RL trainers. If None, auto-resolved
            via resolve_weights_path(). Ignored for fixed-time and max-pressure.
    """
    trainer: str
    topology: str
    n_runs: int = 10
    base_seed: int = 42
    ranked: bool = True
    horizon: int = 450
    weights_path: Optional[str] = None
    use_time_encoding: bool = False


@dataclass
class MCAggregatedResult:
    """Aggregated results from N Monte Carlo trials.

    Attributes:
        config: The MCConfig used to produce these results.
        individual_results: List of per-run metrics as JSON-serializable dicts
            (each from metrics_to_dict()).
        n_completed: Number of runs that completed successfully.
        n_failed: Number of runs that raised exceptions.
        aggregated: Statistical aggregates (mean, std, ci_95, min, max) for
            each key metric.  Keys: avg_waiting_time, avg_travel_time,
            throughput, mean_reward, total_comm_cost.
        errors: Error messages from failed runs (one string per failed run).
    """
    config: MCConfig
    individual_results: List[dict] = field(default_factory=list)
    n_completed: int = 0
    n_failed: int = 0
    aggregated: dict = field(default_factory=dict)
    errors: List[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Internal aggregation helpers
# ---------------------------------------------------------------------------

def _aggregate_metrics(metrics_list: List[TrialMetrics]) -> dict:
    """Compute mean, std, 95% CI, min, max for each key numeric metric.

    95% confidence interval uses the classic formula:
        CI = mean +/- 1.96 * std / sqrt(n)

    All computations use stdlib ``statistics`` and ``math`` — no numpy
    dependency is introduced in this module.

    Args:
        metrics_list: List of TrialMetrics from successful MC runs.
            If empty, returns an empty dict.

    Returns:
        Dict with one entry per key metric.  Each entry is a sub-dict::

            {
                "mean": float,
                "std": float,
                "ci_95_lower": float,
                "ci_95_upper": float,
                "min": float,
                "max": float,
            }
    """
    if not metrics_list:
        return {}

    n = len(metrics_list)

    def _stat_block(values: List[float]) -> dict:
        """Compute statistics for a list of numeric values."""
        mean_val = statistics.mean(values)
        std_val = statistics.stdev(values) if n > 1 else 0.0
        margin = 1.96 * std_val / math.sqrt(n)
        return {
            "mean": round(mean_val, 4),
            "std": round(std_val, 4),
            "ci_95_lower": round(mean_val - margin, 4),
            "ci_95_upper": round(mean_val + margin, 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    avg_waiting_times = [m.tripinfo.avg_waiting_time for m in metrics_list]
    avg_travel_times = [m.tripinfo.avg_travel_time for m in metrics_list]
    throughputs = [m.tripinfo.throughput for m in metrics_list]
    mean_rewards = [m.mean_reward for m in metrics_list]
    total_comm_costs = [float(m.total_comm_cost) for m in metrics_list]

    return {
        "avg_waiting_time": _stat_block(avg_waiting_times),
        "avg_travel_time": _stat_block(avg_travel_times),
        "throughput": _stat_block(throughputs),
        "mean_reward": _stat_block(mean_rewards),
        "total_comm_cost": _stat_block(total_comm_costs),
    }


# ---------------------------------------------------------------------------
# Main public functions
# ---------------------------------------------------------------------------

def run_monte_carlo(
    config: MCConfig,
    on_progress: Optional[Callable] = None,
) -> MCAggregatedResult:
    """Execute N trials for a single (trainer, topology) configuration.

    Seeds are assigned deterministically as ``config.base_seed + i`` for
    ``i in range(config.n_runs)``, matching the SEAL paper's reproducibility
    approach (NUM_MC_RUNS=10, seeds 42..51 by default).

    Failed trials are logged and counted but do not abort the campaign.  The
    aggregation is computed only over successful runs.

    Args:
        config: MCConfig specifying trainer, topology, n_runs, base_seed, etc.
        on_progress: Optional callback invoked after each trial with keyword
            arguments ``completed``, ``total``, ``config``.  Useful for
            streaming progress updates to the frontend.

    Returns:
        MCAggregatedResult with per-run dicts, aggregate statistics, and
        counts of completed vs failed runs.
    """
    # Resolve weights once for all runs (skip for non-RL trainers)
    weights_path = config.weights_path
    if weights_path is None and config.trainer not in ("fixed-time", "max-pressure"):
        try:
            weights_path = resolve_weights_path(config.trainer, config.topology, config.ranked)
        except FileNotFoundError as exc:
            logger.error(
                "Cannot resolve weights for trainer=%s topology=%s: %s",
                config.trainer, config.topology, exc,
            )
            return MCAggregatedResult(
                config=config,
                n_failed=config.n_runs,
                errors=[str(exc)] * config.n_runs,
            )

    individual_results: List[dict] = []
    successful_metrics: List[TrialMetrics] = []
    errors: List[str] = []
    n_completed = 0
    n_failed = 0

    for i in range(config.n_runs):
        seed = config.base_seed + i
        logger.info(
            "MC run %d/%d — trainer=%s topology=%s seed=%d",
            i + 1, config.n_runs, config.trainer, config.topology, seed,
        )

        try:
            trial_result = run_trial(
                trainer_type=config.trainer,
                topology=config.topology,
                seed=seed,
                ranked=config.ranked,
                weights_path=weights_path,
                horizon=config.horizon,
                use_time_encoding=config.use_time_encoding,
            )
            metrics = compute_trial_metrics(trial_result)
            individual_results.append(metrics_to_dict(metrics))
            successful_metrics.append(metrics)
            n_completed += 1
        except Exception as exc:  # noqa: BLE001
            err_msg = (
                f"Run {i + 1} failed (trainer={config.trainer}, "
                f"topology={config.topology}, seed={seed}): {exc}"
            )
            logger.warning(err_msg)
            errors.append(err_msg)
            n_failed += 1

        # Report progress regardless of success/failure
        if on_progress is not None:
            try:
                on_progress(completed=i + 1, total=config.n_runs, config=config)
            except Exception as cb_exc:  # noqa: BLE001
                logger.debug("on_progress callback raised: %s", cb_exc)

    aggregated = _aggregate_metrics(successful_metrics)

    return MCAggregatedResult(
        config=config,
        individual_results=individual_results,
        n_completed=n_completed,
        n_failed=n_failed,
        aggregated=aggregated,
        errors=errors,
    )


def run_full_campaign(
    trainers: Optional[List[str]] = None,
    topologies: Optional[List[str]] = None,
    n_runs: int = 10,
    ranked: bool = True,
    on_progress: Optional[Callable] = None,
) -> List[MCAggregatedResult]:
    """Run the full evaluation campaign across all trainer x topology combinations.

    Default campaign: 5 trainers x 3 topologies x 10 MC runs = 150 total trials.

    Args:
        trainers: List of trainer types to evaluate.  Defaults to all 5
            (TRAINERS: FedRL, MARL, SARL, fixed-time, max-pressure).
        topologies: List of topology names to evaluate.  Defaults to all 3
            (TOPOLOGIES: grid-3x3, grid-5x5, grid-7x7).
        n_runs: Number of MC repetitions per (trainer, topology) pair.
        ranked: Whether to use ranked observation features for RL trainers.
        on_progress: Optional callback forwarded to each run_monte_carlo() call.
            Receives ``completed``, ``total``, ``config`` kwargs.

    Returns:
        List of MCAggregatedResult — one entry per (trainer, topology) pair.
        Up to 15 entries for the default 5x3 campaign.
    """
    if trainers is None:
        trainers = list(TRAINERS)
    if topologies is None:
        topologies = list(TOPOLOGIES)

    campaign_results: List[MCAggregatedResult] = []

    for trainer in trainers:
        for topology in topologies:
            logger.info(
                "Campaign: starting trainer=%s topology=%s n_runs=%d",
                trainer, topology, n_runs,
            )
            config = MCConfig(
                trainer=trainer,
                topology=topology,
                n_runs=n_runs,
                ranked=ranked,
            )
            result = run_monte_carlo(config, on_progress=on_progress)
            campaign_results.append(result)
            logger.info(
                "Campaign: finished trainer=%s topology=%s — %d/%d runs OK",
                trainer, topology, result.n_completed, n_runs,
            )

    return campaign_results


def campaign_to_dict(results: List[MCAggregatedResult]) -> dict:
    """Convert full campaign results to a JSON-serializable dict.

    Output structure::

        {
            "campaign": [
                {
                    "trainer": str,
                    "topology": str,
                    "n_runs": int,
                    "n_completed": int,
                    "n_failed": int,
                    "aggregated": {
                        "avg_waiting_time": {"mean": ..., "std": ..., ...},
                        ...
                    },
                    "individual": [
                        {"trainer": ..., "topology": ..., "mean_reward": ..., ...},
                        ...
                    ],
                    "errors": [...],
                },
                ...
            ]
        }

    Args:
        results: List of MCAggregatedResult from run_full_campaign() or
            multiple run_monte_carlo() calls.

    Returns:
        Plain dict suitable for json.dumps() or FastAPI response bodies.
    """
    campaign_entries = []
    for r in results:
        campaign_entries.append({
            "trainer": r.config.trainer,
            "topology": r.config.topology,
            "n_runs": r.config.n_runs,
            "n_completed": r.n_completed,
            "n_failed": r.n_failed,
            "aggregated": r.aggregated,
            "individual": r.individual_results,
            "errors": r.errors,
        })
    return {"campaign": campaign_entries}
