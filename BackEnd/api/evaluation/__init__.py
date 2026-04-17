"""SEAL Evaluation Framework.

Provides evaluation campaigns, metrics computation, cross-topology transfer
testing, Monte Carlo orchestration, and persistent result storage.
"""

from .runner import run_trial, TrialResult, resolve_weights_path, TRAINERS, TOPOLOGIES
from .baselines import run_fixed_time_trial, run_max_pressure_trial, run_rl_trial
from .metrics import (
    compute_tripinfo_metrics, compute_trial_metrics,
    TripinfoMetrics, TrialMetrics, metrics_to_dict,
)
from .transfer import (
    build_transfer_matrix, compute_transfer_gap,
    TransferResult, transfer_matrix_to_dict,
)
from .monte_carlo import (
    run_monte_carlo, run_full_campaign,
    MCConfig, MCAggregatedResult, campaign_to_dict,
)
from .store import save_evaluation, load_evaluation, list_evaluations

__all__ = [
    "run_trial", "TrialResult", "resolve_weights_path", "TRAINERS", "TOPOLOGIES",
    "run_fixed_time_trial", "run_max_pressure_trial", "run_rl_trial",
    "compute_tripinfo_metrics", "compute_trial_metrics",
    "TripinfoMetrics", "TrialMetrics", "metrics_to_dict",
    "build_transfer_matrix", "compute_transfer_gap",
    "TransferResult", "transfer_matrix_to_dict",
    "run_monte_carlo", "run_full_campaign",
    "MCConfig", "MCAggregatedResult", "campaign_to_dict",
    "save_evaluation", "load_evaluation", "list_evaluations",
]
