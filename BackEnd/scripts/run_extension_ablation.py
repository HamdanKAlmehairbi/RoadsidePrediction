"""CLI script for running SEAL extension ablation studies.

Runs three parameterized ablation studies for the Phase 4 extensions:
  - FedProx: compare mu in {0.0, 0.01, 0.1}
  - Cooperative reward: compare alpha in {1.0, 0.5, 0.1}
  - Time-of-day: compare fixed demand vs. time-of-day curriculum + encoding

Each ablation trains fresh weights then evaluates with 10 MC seeds to produce
statistically comparable results. Results are saved per-ablation as named
campaign directories under BackEnd/results/campaigns/.

Usage:
    # Smoke test — 1 seed per config
    cd BackEnd && python scripts/run_extension_ablation.py --ablation fedprox --dry-run 1

    # Full FedProx ablation (10 seeds per config)
    cd BackEnd && python scripts/run_extension_ablation.py --ablation fedprox

    # Run all three ablations sequentially
    cd BackEnd && python scripts/run_extension_ablation.py --ablation all

    # Override topology and episodes
    cd BackEnd && python scripts/run_extension_ablation.py --ablation cooperative \\
        --topology grid-5x5 --n-episodes 100

IMPORTANT: This script imports Python modules directly (NOT via REST API).
It does NOT start the FastAPI server. All imports are direct:
    from scripts.run_campaign import train_and_evaluate, save_campaign_results
    from api.evaluation.campaign_config import ExtensionConfig
"""
import argparse
import json
import logging
import os
import sys
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure BackEnd/ is on sys.path so `api.*` and `scripts.*` imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import CampaignResult, ExtensionConfig
from scripts.run_campaign import save_campaign_results, train_and_evaluate

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Ablation config builders
# ---------------------------------------------------------------------------


def build_fedprox_configs(
    topology: str = "grid-3x3",
    n_episodes: int = 50,
    n_eval_runs: int = 10,
) -> List[ExtensionConfig]:
    """Build FedProx ablation configs: mu in {0.0, 0.01, 0.1}.

    mu=0.0 is FedAvg baseline (no proximal term).
    mu=0.01 and mu=0.1 activate the FedProx proximal loss.

    Args:
        topology: SUMO network topology, e.g. "grid-3x3".
        n_episodes: Training episodes for each config.
        n_eval_runs: Monte Carlo evaluation runs for each config.

    Returns:
        List of 3 ExtensionConfigs covering the FedProx mu sweep.
    """
    return [
        ExtensionConfig(
            name="fedavg_baseline",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fedprox_mu=0.0,
        ),
        ExtensionConfig(
            name="fedprox_mu0.01",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fedprox_mu=0.01,
        ),
        ExtensionConfig(
            name="fedprox_mu0.1",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fedprox_mu=0.1,
        ),
    ]


def build_cooperative_configs(
    topology: str = "grid-3x3",
    n_episodes: int = 50,
    n_eval_runs: int = 10,
) -> List[ExtensionConfig]:
    """Build cooperative reward ablation configs: alpha in {1.0, 0.5, 0.1}.

    alpha=1.0 is the selfish baseline (each agent maximises only its own reward).
    alpha=0.5 blends local and neighbor rewards equally.
    alpha=0.1 weights neighbor rewards heavily (near-fully cooperative).

    # alpha=0.0 excluded — creates degenerate zero reward signal (see research pitfall 6)

    Args:
        topology: SUMO network topology, e.g. "grid-3x3".
        n_episodes: Training episodes for each config.
        n_eval_runs: Monte Carlo evaluation runs for each config.

    Returns:
        List of 3 ExtensionConfigs covering the cooperative alpha sweep.
    """
    return [
        ExtensionConfig(
            name="selfish_baseline",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            alpha=1.0,
        ),
        ExtensionConfig(
            name="cooperative_alpha0.5",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            alpha=0.5,
        ),
        ExtensionConfig(
            name="cooperative_alpha0.1",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            alpha=0.1,
        ),
    ]


def build_time_of_day_configs(
    topology: str = "grid-3x3",
    n_episodes: int = 50,
    n_eval_runs: int = 10,
) -> List[ExtensionConfig]:
    """Build time-of-day ablation configs: fixed demand vs. ToD curriculum + encoding.

    config[0]: Fixed demand — no time-of-day variation, no time encoding.
    config[1]: ToD with encoding — demand varies by time-of-day AND the observation
               includes sin/cos time features (indices 14-15 in ranked observation).

    Note: time_of_day=True without use_time_encoding=True is a valid intermediate
    but not studied here — we pair them as the natural "full ToD" treatment.

    Args:
        topology: SUMO network topology, e.g. "grid-3x3".
        n_episodes: Training episodes for each config.
        n_eval_runs: Monte Carlo evaluation runs for each config.

    Returns:
        List of 2 ExtensionConfigs: fixed demand baseline and full ToD treatment.
    """
    return [
        ExtensionConfig(
            name="fixed_demand",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            time_of_day=False,
            use_time_encoding=False,
        ),
        ExtensionConfig(
            name="tod_no_encoding",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            time_of_day=True,
            use_time_encoding=False,
        ),
        ExtensionConfig(
            name="tod_with_encoding",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            time_of_day=True,
            use_time_encoding=True,
        ),
    ]


def build_strategy_configs(
    topology: str = "grid-3x3",
    n_episodes: int = 50,
    n_eval_runs: int = 10,
) -> List[ExtensionConfig]:
    """Core 10-way strategy comparison: the full spectrum from independent to shared.

    MARL        — fully independent, no sharing
    MeanField   — independent + neighbor action in obs (approximate interaction)
    CTDE        — independent actors, shared critic (global training signal)
    Gossip      — peer-to-peer neighbor weight averaging (decentralized mesh)
    HierFed     — two-tier cluster-then-global weight averaging (tree topology)
    FedDistill  — share action logits not weights (knowledge distillation)
    FedRL       — central server weight averaging (star topology)
    SARL        — one shared policy (full sharing)
    fixed-time  — non-RL floor baseline
    max-pressure — non-RL floor baseline
    """
    return [
        ExtensionConfig(
            name="marl",
            trainer_type="MARL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="mean_field",
            trainer_type="MeanField",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="ctde",
            trainer_type="CTDE",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="gossip",
            trainer_type="Gossip",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="hierfed",
            trainer_type="HierFed",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="feddistill",
            trainer_type="FedDistill",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="fedrl",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="sarl",
            trainer_type="SARL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="fixed_time",
            trainer_type="fixed-time",
            topology=topology,
            n_episodes=0,
            n_eval_runs=n_eval_runs,
            weights_path="__baseline__",
        ),
        ExtensionConfig(
            name="max_pressure",
            trainer_type="max-pressure",
            topology=topology,
            n_episodes=0,
            n_eval_runs=n_eval_runs,
            weights_path="__baseline__",
        ),
    ]


def build_aggregation_configs(
    topology: str = "grid-3x3",
    n_episodes: int = 50,
    n_eval_runs: int = 10,
) -> List[ExtensionConfig]:
    """Pure aggregation strategy ablation: naive vs reward-weighted vs traffic-weighted.

    All use FedRL with mu=0.0 — only the aggregation weighting rule changes.
    FedProx is excluded here (it changes the loss function, not aggregation).
    """
    return [
        ExtensionConfig(
            name="aggr_naive",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            aggr="naive",
            fedprox_mu=0.0,
        ),
        ExtensionConfig(
            name="aggr_reward_weighted",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            aggr="pos_reward",
            fedprox_mu=0.0,
        ),
        ExtensionConfig(
            name="aggr_traffic_weighted",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            aggr="traffic",
            fedprox_mu=0.0,
        ),
    ]


def build_fedrl_variant_configs(
    topology: str = "grid-3x3",
    n_episodes: int = 50,
    n_eval_runs: int = 10,
) -> List[ExtensionConfig]:
    """FedRL variant ablation: standard vs clustered vs partial vs soft-update.

    Tests how different federation strategies affect per-intersection specialization
    and convergence.
    """
    return [
        ExtensionConfig(
            name="fedrl_standard",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
        ),
        ExtensionConfig(
            name="fedrl_reward_grouped",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fed_cluster=True,
        ),
        ExtensionConfig(
            name="fedrl_partial",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fed_partial=True,
        ),
        ExtensionConfig(
            name="fedrl_soft_tau0.5",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fed_tau=0.5,
        ),
    ]


# ---------------------------------------------------------------------------
# Ablation runner
# ---------------------------------------------------------------------------


def _load_completed_names(ablation_name: str) -> set:
    """Load config names already saved in results.json for this ablation."""
    _BACKEND_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    results_path = os.path.join(
        _BACKEND_DIR, "results", "campaigns", ablation_name, "results.json"
    )
    completed = set()
    if os.path.exists(results_path):
        try:
            with open(results_path, "r", encoding="utf-8") as fp:
                data = json.load(fp)
                for r in data.get("results", []):
                    name = r.get("config", {}).get("name")
                    if name and not r.get("error"):
                        completed.add(name)
        except (json.JSONDecodeError, KeyError):
            pass
    return completed


def run_ablation(
    configs: List[ExtensionConfig],
    ablation_name: str,
    dry_run_seeds: Optional[int] = None,
) -> List[CampaignResult]:
    """Run a single ablation study: train + evaluate each config, then save results.

    Supports resumption — skips configs whose names already appear (without
    error) in the existing results.json for this ablation. Saves incrementally
    after each config so progress survives crashes.

    Args:
        configs: List of ExtensionConfig — each represents one condition.
        ablation_name: Directory name for output (e.g. "fedprox-ablation").
        dry_run_seeds: If set, override n_eval_runs on each config to this value.
            Useful for smoke testing before committing to full runs.

    Returns:
        List of CampaignResult — one per config, in order.
    """
    if dry_run_seeds is not None:
        for cfg in configs:
            cfg.n_eval_runs = dry_run_seeds
        logger.info(
            "Ablation %s: dry-run mode — overriding n_eval_runs to %d",
            ablation_name, dry_run_seeds,
        )

    completed_names = _load_completed_names(ablation_name)
    if completed_names:
        logger.info(
            "Ablation %s: resuming — %d configs already completed: %s",
            ablation_name, len(completed_names), ", ".join(sorted(completed_names)),
        )

    results: List[CampaignResult] = []
    total = len(configs)

    for i, config in enumerate(configs):
        if config.name in completed_names:
            logger.info(
                "Ablation %s: SKIPPING '%s' (%d/%d) — already completed",
                ablation_name, config.name, i + 1, total,
            )
            continue

        logger.info(
            "Ablation %s: running config '%s' (%d/%d)",
            ablation_name, config.name, i + 1, total,
        )
        result = train_and_evaluate(config)
        results.append(result)
        logger.info(
            "Ablation %s: config '%s' done — %.1fs, error=%s",
            ablation_name, config.name, result.duration_seconds, result.error,
        )

        # Save incrementally so progress survives crashes
        save_campaign_results([result], ablation_name)

    return results


# ---------------------------------------------------------------------------
# Summary printer
# ---------------------------------------------------------------------------


def _print_summary_table(all_results: List[CampaignResult]) -> None:
    """Print a summary table of ablation results to stdout.

    Columns: config name, n_completed, n_failed, avg_waiting_time mean,
    duration_seconds.
    """
    header = (
        f"{'Config':<30} {'Completed':>9} {'Failed':>6} "
        f"{'AvgWait (mean)':>14} {'Duration(s)':>11}"
    )
    print("\n" + "=" * 76)
    print("Ablation Summary")
    print("=" * 76)
    print(header)
    print("-" * 76)

    for r in all_results:
        n_completed = r.evaluation.n_completed if r.evaluation else 0
        n_failed = (r.evaluation.n_failed if r.evaluation else 0) + (
            1 if r.error else 0
        )
        # avg_waiting_time mean — available on MCAggregatedResult if present
        avg_wait = "N/A"
        if r.evaluation is not None:
            agg = r.evaluation
            if hasattr(agg, "avg_waiting_time") and agg.avg_waiting_time is not None:
                wt = agg.avg_waiting_time
                if hasattr(wt, "mean"):
                    avg_wait = f"{wt.mean:.2f}"
                elif isinstance(wt, (int, float)):
                    avg_wait = f"{wt:.2f}"

        print(
            f"{r.config.name:<30} {n_completed:>9} {n_failed:>6} "
            f"{avg_wait:>14} {r.duration_seconds:>11.1f}"
        )

    print("=" * 76)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch to the requested ablation study."""
    parser = argparse.ArgumentParser(
        description=(
            "SEAL extension ablation runner — parameterised experiments for "
            "FedProx, cooperative reward, and time-of-day."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--ablation",
        choices=[
            "fedprox", "cooperative", "time-of-day",
            "strategy", "aggregation", "fedrl-variants",
            "all", "full",
        ],
        required=True,
        help=(
            "Which ablation to run: 'fedprox' (mu sweep), 'cooperative' (alpha sweep), "
            "'time-of-day' (fixed vs. ToD+encoding), 'strategy' (FedRL vs MARL vs SARL), "
            "'aggregation' (naive vs reward-weighted vs FedProx), "
            "'fedrl-variants' (standard vs clustered vs partial vs soft-update), "
            "'all' (original 3 ablations), or 'full' (all 6 experiments)."
        ),
    )
    parser.add_argument(
        "--topology",
        default="grid-3x3",
        help="SUMO network topology (e.g. grid-3x3, grid-5x5).",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=None,
        help="SUMO network topologies (e.g. grid-3x3 grid-5x5). Overrides --topology.",
    )
    parser.add_argument(
        "--n-episodes",
        type=int,
        default=50,
        dest="n_episodes",
        help="Number of training episodes per config.",
    )
    parser.add_argument(
        "--n-eval-runs",
        type=int,
        default=10,
        dest="n_eval_runs",
        help="Number of Monte Carlo evaluation seeds per config.",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N_SEEDS",
        dest="dry_run",
        help=(
            "If set, override n_eval_runs to N_SEEDS for smoke testing "
            "(e.g. --dry-run 1 for a minimal end-to-end check)."
        ),
    )

    parser.add_argument(
        "--training-seeds",
        nargs="+",
        type=int,
        default=[54321],
        dest="training_seeds",
        help="Training seeds to run per config (e.g. --training-seeds 42 123 456)",
    )
    parser.add_argument(
        "--demand-levels",
        nargs="+",
        type=int,
        default=[360],
        dest="demand_levels",
        help="VPLPH demand levels to test (e.g. --demand-levels 150 360 600)",
    )

    args = parser.parse_args()

    all_results: List[CampaignResult] = []

    ALL_ORIGINAL = ["fedprox", "cooperative", "time-of-day"]
    ALL_NEW = ["strategy", "aggregation", "fedrl-variants"]
    if args.ablation == "all":
        ablations_to_run = ALL_ORIGINAL
    elif args.ablation == "full":
        ablations_to_run = ALL_ORIGINAL + ALL_NEW
    else:
        ablations_to_run = [args.ablation]

    # --topologies takes precedence over --topology
    topologies = args.topologies if args.topologies else [args.topology]

    campaign_name_map = {
        "fedprox": "fedprox-ablation",
        "cooperative": "cooperative-ablation",
        "time-of-day": "tod-ablation",
        "strategy": "strategy-comparison",
        "aggregation": "aggregation-ablation",
        "fedrl-variants": "fedrl-variants",
    }

    import copy

    for topology in topologies:
        logger.info("=== Topology: %s ===", topology)
        for demand in args.demand_levels:
            for ablation in ablations_to_run:
                logger.info(
                    "=== Starting ablation: %s (topology=%s, demand=%d) ===",
                    ablation, topology, demand,
                )

                builder_map = {
                    "fedprox": build_fedprox_configs,
                    "cooperative": build_cooperative_configs,
                    "time-of-day": build_time_of_day_configs,
                    "strategy": build_strategy_configs,
                    "aggregation": build_aggregation_configs,
                    "fedrl-variants": build_fedrl_variant_configs,
                }
                configs = builder_map[ablation](
                    topology=topology,
                    n_episodes=args.n_episodes,
                    n_eval_runs=args.n_eval_runs,
                )

                # Set demand level on all configs
                for cfg in configs:
                    cfg.vplph = demand

                # Expand configs across training seeds
                expanded = []
                for cfg in configs:
                    for seed in args.training_seeds:
                        c = copy.deepcopy(cfg)
                        c.training_seed = seed
                        if len(args.training_seeds) > 1 or len(args.demand_levels) > 1:
                            suffix = ""
                            if len(args.training_seeds) > 1:
                                suffix += f"_s{seed}"
                            if len(args.demand_levels) > 1:
                                suffix += f"_d{demand}"
                            c.name = c.name + suffix
                        expanded.append(c)

                base_campaign = campaign_name_map[ablation]
                campaign_name = f"{base_campaign}/{topology}"
                results = run_ablation(
                    configs=expanded,
                    ablation_name=campaign_name,
                    dry_run_seeds=args.dry_run,
                )
                all_results.extend(results)
                logger.info("=== Ablation %s complete: %d configs ===", ablation, len(results))

    _print_summary_table(all_results)


if __name__ == "__main__":
    main()
