"""Metrics computation for SEAL evaluation.

Provides two computation paths:
- compute_tripinfo_metrics(): parse SUMO tripinfo XML for traffic KPIs
- compute_trial_metrics(): combine tripinfo with RL-level reward and comm cost data

Usage:
    from api.evaluation.metrics import (
        compute_tripinfo_metrics,
        compute_trial_metrics,
        TripinfoMetrics,
        TrialMetrics,
        metrics_to_dict,
    )

    tripinfo = compute_tripinfo_metrics("path/to/tripinfo.xml")
    print(tripinfo.avg_waiting_time, tripinfo.throughput)

    trial_metrics = compute_trial_metrics(trial_result)
    print(trial_metrics.mean_reward, trial_metrics.comm_costs)
"""
import logging
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from typing import Dict, List

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data classes
# ---------------------------------------------------------------------------

@dataclass
class TripinfoMetrics:
    """Traffic KPIs extracted from a SUMO tripinfo XML file.

    Attributes:
        avg_waiting_time: Mean waiting time across all completed trips (seconds).
        avg_travel_time: Mean total travel time across all completed trips (seconds).
        avg_depart_delay: Mean departure delay across all completed trips (seconds).
        throughput: completed_trips / spawned_trips ratio (0.0–1.0).
            When the spawned count is unavailable, defaults to 1.0 (all recorded
            trips are treated as completed).
        completed_trips: Number of tripinfo entries (= vehicles that completed).
        total_waiting_time: Sum of all waitingTime values (seconds).
        total_travel_time: Sum of all duration values (seconds).
    """
    avg_waiting_time: float = 0.0
    avg_travel_time: float = 0.0
    avg_depart_delay: float = 0.0
    throughput: float = 0.0
    completed_trips: int = 0
    total_waiting_time: float = 0.0
    total_travel_time: float = 0.0


@dataclass
class TrialMetrics:
    """Combined metrics for a single evaluation trial.

    Attributes:
        tripinfo: Traffic KPIs from the SUMO tripinfo XML.
        mean_reward: Mean of all per-TLS per-step rewards.
        total_reward: Sum of all rewards.
        comm_costs: Communication cost by type (message counts per comm type).
        total_comm_cost: Sum of all values in comm_costs.
        trainer: Trainer type ("FedRL", "MARL", "SARL", "fixed-time", "max-pressure").
        topology: Topology name (e.g. "grid-3x3").
        seed: Random seed used for route generation.
    """
    tripinfo: TripinfoMetrics = field(default_factory=TripinfoMetrics)
    mean_reward: float = 0.0
    total_reward: float = 0.0
    comm_costs: Dict[str, int] = field(default_factory=dict)
    total_comm_cost: int = 0
    trainer: str = ""
    topology: str = ""
    seed: int = 0


# ---------------------------------------------------------------------------
# Computation helpers
# ---------------------------------------------------------------------------

def compute_tripinfo_metrics(tripinfo_path: str) -> TripinfoMetrics:
    """Parse a SUMO tripinfo XML file and compute traffic KPIs.

    Args:
        tripinfo_path: Absolute or relative path to a SUMO tripinfo XML file.

    Returns:
        TripinfoMetrics populated from the XML data.  Returns an all-zero
        TripinfoMetrics if the file is missing or malformed (with a warning).
    """
    try:
        tree = ET.parse(tripinfo_path)
    except FileNotFoundError:
        logger.warning("Tripinfo file not found: %s", tripinfo_path)
        return TripinfoMetrics()
    except ET.ParseError as exc:
        logger.warning("Failed to parse tripinfo XML %s: %s", tripinfo_path, exc)
        return TripinfoMetrics()

    root = tree.getroot()
    trips = root.findall("tripinfo")

    if not trips:
        logger.warning("No <tripinfo> elements found in %s", tripinfo_path)
        return TripinfoMetrics()

    waiting_times: List[float] = []
    travel_times: List[float] = []
    depart_delays: List[float] = []

    for trip in trips:
        waiting_times.append(float(trip.get("waitingTime", 0)))
        travel_times.append(float(trip.get("duration", 0)))
        depart_delays.append(float(trip.get("departDelay", 0)))

    completed_trips = len(trips)
    total_waiting = sum(waiting_times)
    total_travel = sum(travel_times)

    # Throughput: without a spawned count we treat all recorded trips as
    # completed (ratio = 1.0).  Monte Carlo aggregation can override this
    # once the route file vehicle count is available.
    throughput = 1.0

    return TripinfoMetrics(
        avg_waiting_time=total_waiting / completed_trips,
        avg_travel_time=total_travel / completed_trips,
        avg_depart_delay=sum(depart_delays) / completed_trips,
        throughput=throughput,
        completed_trips=completed_trips,
        total_waiting_time=total_waiting,
        total_travel_time=total_travel,
    )


def compute_trial_metrics(trial_result) -> TrialMetrics:
    """Compute TrialMetrics from a completed TrialResult.

    Args:
        trial_result: A TrialResult instance (from api.evaluation.runner).

    Returns:
        TrialMetrics combining tripinfo KPIs with reward and comm cost data.
    """
    from .runner import TrialResult  # local import to avoid circular deps at module level

    # --- Reward aggregation ---
    all_rewards: List[float] = []
    for rewards_list in trial_result.per_tls_rewards.values():
        all_rewards.extend(rewards_list)

    mean_reward = (sum(all_rewards) / len(all_rewards)) if all_rewards else 0.0
    total_reward = sum(all_rewards)

    # --- Comm costs ---
    comm_costs: Dict[str, int] = dict(trial_result.comm_costs)
    total_comm_cost = sum(comm_costs.values())

    # --- Tripinfo ---
    tripinfo = compute_tripinfo_metrics(trial_result.tripinfo_path)

    # Override throughput with real value: completed / spawned
    if trial_result.n_vehicles_total > 0:
        tripinfo.throughput = tripinfo.completed_trips / trial_result.n_vehicles_total

    return TrialMetrics(
        tripinfo=tripinfo,
        mean_reward=mean_reward,
        total_reward=total_reward,
        comm_costs=comm_costs,
        total_comm_cost=total_comm_cost,
        trainer=trial_result.trainer,
        topology=trial_result.topology,
        seed=trial_result.seed,
    )


def metrics_to_dict(metrics: TrialMetrics) -> dict:
    """Convert a TrialMetrics instance to a JSON-serializable dict.

    All float values are rounded to 4 decimal places for compact storage.

    Args:
        metrics: TrialMetrics instance to convert.

    Returns:
        A plain dict suitable for json.dumps() or API responses.
    """
    def _r(v: float) -> float:
        return round(v, 4)

    tripinfo = metrics.tripinfo
    return {
        "trainer": metrics.trainer,
        "topology": metrics.topology,
        "seed": metrics.seed,
        "mean_reward": _r(metrics.mean_reward),
        "total_reward": _r(metrics.total_reward),
        "comm_costs": metrics.comm_costs,
        "total_comm_cost": metrics.total_comm_cost,
        "tripinfo": {
            "avg_waiting_time": _r(tripinfo.avg_waiting_time),
            "avg_travel_time": _r(tripinfo.avg_travel_time),
            "avg_depart_delay": _r(tripinfo.avg_depart_delay),
            "throughput": _r(tripinfo.throughput),
            "completed_trips": tripinfo.completed_trips,
            "total_waiting_time": _r(tripinfo.total_waiting_time),
            "total_travel_time": _r(tripinfo.total_travel_time),
        },
    }
