"""Standalone CLI script for running SEAL experiment campaigns.

Reproduces SEAL paper baseline results by running train-then-evaluate
pipelines for FedRL, MARL, SARL (using example weights), plus fixed-time
and max-pressure baselines.

Usage:
    # Dry run (1 seed per config) for smoke testing
    cd BackEnd && python scripts/run_campaign.py --campaign baseline --dry-run 1

    # Full baseline campaign (10 seeds per config)
    cd BackEnd && python scripts/run_campaign.py --campaign baseline

    # Custom topologies and output name
    cd BackEnd && python scripts/run_campaign.py --campaign baseline \\
        --topologies grid-3x3 --n-eval-runs 5 --output-name baseline_3x3

IMPORTANT: This script imports Python modules directly (NOT via REST API).
It does NOT start the FastAPI server. All imports are direct:
    from api.training_runner import create_trainer, run_training_loop
    from api.evaluation.monte_carlo import run_monte_carlo, MCConfig
"""
import argparse
import json
import logging
import os
import sys
import time
from typing import List, Optional

# ---------------------------------------------------------------------------
# Ensure BackEnd/ is on sys.path so `api.*` imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import (
    CampaignResult,
    ExtensionConfig,
    config_to_dict,
    resolve_example_weights,
    result_to_dict,
)
from api.evaluation.monte_carlo import MCConfig, run_monte_carlo
from api.training_runner import create_trainer, run_training_loop

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Core campaign functions
# ---------------------------------------------------------------------------


def train_and_evaluate(
    config: ExtensionConfig,
    on_progress=None,
) -> CampaignResult:
    """Run a full train-then-evaluate campaign for a single ExtensionConfig.

    If config.weights_path is set, the training step is skipped and the
    provided weights are used directly for Monte Carlo evaluation.

    Args:
        config: ExtensionConfig specifying all experiment parameters.
        on_progress: Optional callback forwarded to run_monte_carlo().
            Called after each MC trial with (completed, total, config).

    Returns:
        CampaignResult with training rewards, evaluation metrics, weights path,
        error message (if failed), and duration in seconds.
    """
    start_time = time.time()
    result = CampaignResult(config=config)

    try:
        weights_path = config.weights_path

        # --- Training step (skip if weights provided) ---
        if weights_path is None:
            logger.info(
                "Training %s on %s (%d episodes) ...",
                config.trainer_type, config.topology, config.n_episodes,
            )
            trainer = create_trainer(
                trainer_type=config.trainer_type,
                topology=config.topology,
                ranked=config.ranked,
                n_episodes=config.n_episodes,
                fedprox_mu=config.fedprox_mu,
                alpha=config.alpha,
                time_of_day=config.time_of_day,
                use_time_encoding=config.use_time_encoding,
                vplph=config.vplph,
                aggr=config.aggr,
                fed_tau=config.fed_tau,
                fed_cluster=config.fed_cluster,
                fed_partial=config.fed_partial,
            )
            training_output = run_training_loop(trainer, n_episodes=config.n_episodes)
            result.training_rewards = training_output.get("rewards", [])
            weights_path = training_output.get("weights_path")
            logger.info("Training complete. Weights: %s", weights_path)
        else:
            logger.info(
                "Skipping training — using provided weights: %s", weights_path
            )
            result.training_rewards = None

        result.weights_path = weights_path

        # --- Evaluation step ---
        logger.info(
            "Evaluating %s on %s (%d MC runs) ...",
            config.trainer_type, config.topology, config.n_eval_runs,
        )
        mc_config = MCConfig(
            trainer=config.trainer_type,
            topology=config.topology,
            n_runs=config.n_eval_runs,
            base_seed=config.base_seed,
            ranked=config.ranked,
            horizon=config.horizon,
            weights_path=weights_path,
            use_time_encoding=config.use_time_encoding,
            vplph=config.vplph,
        )
        mc_result = run_monte_carlo(mc_config, on_progress=on_progress)
        result.evaluation = mc_result
        logger.info(
            "Evaluation complete — %d/%d runs OK",
            mc_result.n_completed, config.n_eval_runs,
        )

    except Exception as exc:  # noqa: BLE001
        logger.error(
            "Campaign failed for %s on %s: %s",
            config.trainer_type, config.topology, exc,
        )
        result.error = str(exc)

    result.duration_seconds = round(time.time() - start_time, 2)
    return result


def run_baseline_campaign(
    topologies: Optional[List[str]] = None,
    n_eval_runs: int = 10,
    dry_run_seeds: Optional[int] = None,
) -> List[CampaignResult]:
    """Run the SEAL paper baseline reproduction campaign.

    Evaluates FedRL, MARL, SARL (using example weights — no training),
    plus fixed-time and max-pressure baselines. Uses paper-original weights
    from example_weights/ICCPS/Final/ for RL trainers.

    Args:
        topologies: List of topology names. Defaults to ["grid-3x3", "grid-5x5"].
            grid-7x7 is excluded by default — no example weights available.
        n_eval_runs: Number of MC evaluation runs per config.
        dry_run_seeds: If set (e.g. 2), override n_eval_runs to this value.
            Useful for quick smoke testing (--dry-run 1).

    Returns:
        List of CampaignResult — one per (trainer, topology) combination.
    """
    if topologies is None:
        topologies = ["grid-3x3", "grid-5x5"]

    effective_n_runs = dry_run_seeds if dry_run_seeds is not None else n_eval_runs

    rl_trainers = ["FedRL", "MARL", "SARL"]
    baseline_trainers = ["fixed-time", "max-pressure"]

    configs: List[ExtensionConfig] = []

    # RL trainers — use example weights (no training needed)
    for trainer in rl_trainers:
        for topology in topologies:
            weights_path = resolve_example_weights(trainer, topology)
            if weights_path is None:
                logger.warning(
                    "No example weights for %s on %s — skipping",
                    trainer, topology,
                )
                continue
            cfg = ExtensionConfig(
                name=f"{trainer.lower()}_{topology}_baseline",
                trainer_type=trainer,
                topology=topology,
                n_eval_runs=effective_n_runs,
                weights_path=weights_path,
            )
            configs.append(cfg)

    # Non-RL baselines — no weights needed, MCConfig handles them directly
    for trainer in baseline_trainers:
        for topology in topologies:
            cfg = ExtensionConfig(
                name=f"{trainer.replace('-', '_')}_{topology}_baseline",
                trainer_type=trainer,
                topology=topology,
                n_eval_runs=effective_n_runs,
                weights_path="__baseline__",  # sentinel: evaluated via MCConfig directly
            )
            configs.append(cfg)

    results: List[CampaignResult] = []
    total = len(configs)

    for i, cfg in enumerate(configs, start=1):
        logger.info(
            "Campaign %d/%d: %s on %s — starting %d MC runs",
            i, total, cfg.trainer_type, cfg.topology, cfg.n_eval_runs,
        )

        # Non-RL baselines: create MCConfig directly (no weights needed)
        if cfg.trainer_type in ("fixed-time", "max-pressure"):
            start_time = time.time()
            camp_result = CampaignResult(config=cfg)
            try:
                mc_config = MCConfig(
                    trainer=cfg.trainer_type,
                    topology=cfg.topology,
                    n_runs=cfg.n_eval_runs,
                    base_seed=cfg.base_seed,
                    ranked=cfg.ranked,
                    horizon=cfg.horizon,
                    weights_path=None,  # non-RL — no weights
                )
                mc_result = run_monte_carlo(mc_config)
                camp_result.evaluation = mc_result
                logger.info(
                    "Campaign %d/%d: %s on %s — %d/%d runs OK",
                    i, total, cfg.trainer_type, cfg.topology,
                    mc_result.n_completed, cfg.n_eval_runs,
                )
            except Exception as exc:  # noqa: BLE001
                logger.error(
                    "Campaign %d/%d failed: %s on %s: %s",
                    i, total, cfg.trainer_type, cfg.topology, exc,
                )
                camp_result.error = str(exc)
            camp_result.duration_seconds = round(time.time() - start_time, 2)
            results.append(camp_result)
        else:
            # RL trainers with pre-loaded weights
            camp_result = train_and_evaluate(cfg)
            logger.info(
                "Campaign %d/%d: %s on %s — done (%.1fs)",
                i, total, cfg.trainer_type, cfg.topology,
                camp_result.duration_seconds,
            )
            results.append(camp_result)

    return results


def save_campaign_results(
    results: List[CampaignResult],
    campaign_name: str,
) -> str:
    """Save campaign results as JSON files in BackEnd/results/campaigns/.

    Creates two files:
    - results.json: All CampaignResult data serialized via result_to_dict()
    - config.json: List of ExtensionConfig dicts used in the campaign

    Args:
        results: List of CampaignResult from run_baseline_campaign() or
            a series of train_and_evaluate() calls.
        campaign_name: Name for the output directory (e.g. "baseline").

    Returns:
        Absolute path to the campaign output directory.
    """
    _BACKEND_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    output_dir = os.path.join(_BACKEND_DIR, "results", "campaigns", campaign_name)
    os.makedirs(output_dir, exist_ok=True)

    # Save results.json
    results_data = [result_to_dict(r) for r in results]
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump({"results": results_data}, fp, default=str, indent=2)
    logger.info("Saved %d campaign results to %s", len(results), results_path)

    # Save config.json
    configs_data = [config_to_dict(r.config) for r in results]
    config_path = os.path.join(output_dir, "config.json")
    with open(config_path, "w", encoding="utf-8") as fp:
        json.dump(configs_data, fp, default=str, indent=2)
    logger.info("Saved campaign configs to %s", config_path)

    return output_dir


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------


def main():
    """Parse CLI arguments and run the requested campaign."""
    parser = argparse.ArgumentParser(
        description="SEAL experiment campaign runner — reproduce paper baseline results.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--campaign",
        choices=["baseline", "custom"],
        default="baseline",
        help="Campaign type: 'baseline' runs paper reproduction (FedRL/MARL/SARL + baselines)",
    )
    parser.add_argument(
        "--topologies",
        nargs="+",
        default=["grid-3x3", "grid-5x5"],
        metavar="TOPOLOGY",
        help="Topologies to evaluate (space-separated)",
    )
    parser.add_argument(
        "--n-eval-runs",
        type=int,
        default=10,
        dest="n_eval_runs",
        help="Number of Monte Carlo evaluation runs per config",
    )
    parser.add_argument(
        "--dry-run",
        type=int,
        default=None,
        metavar="N_SEEDS",
        dest="dry_run",
        help="If set, run only N seeds per config (e.g. --dry-run 2 for smoke testing)",
    )
    parser.add_argument(
        "--output-name",
        default="baseline",
        dest="output_name",
        help="Name for the output directory under BackEnd/results/campaigns/",
    )

    args = parser.parse_args()

    if args.campaign == "custom":
        parser.error(
            "Custom campaigns not yet implemented. "
            "Use --campaign baseline for paper reproduction."
        )

    logger.info(
        "Starting %s campaign — topologies=%s, n_eval_runs=%d, dry_run=%s",
        args.campaign, args.topologies, args.n_eval_runs, args.dry_run,
    )

    results = run_baseline_campaign(
        topologies=args.topologies,
        n_eval_runs=args.n_eval_runs,
        dry_run_seeds=args.dry_run,
    )

    output_dir = save_campaign_results(results, campaign_name=args.output_name)

    # Print summary
    n_completed = sum(
        (r.evaluation.n_completed if r.evaluation else 0) for r in results
    )
    n_failed = sum(
        (r.evaluation.n_failed if r.evaluation else 0) + (1 if r.error else 0)
        for r in results
    )
    print("\n" + "=" * 60)
    print(f"Campaign complete: {len(results)} configs run")
    print(f"  Total MC runs completed: {n_completed}")
    print(f"  Total MC runs failed:    {n_failed}")
    print(f"  Output directory:        {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
