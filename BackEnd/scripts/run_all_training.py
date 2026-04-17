"""Run all missing training experiments with incremental saving.

Each config saves immediately after completing, so progress is never lost.
Results are saved to results/campaigns/training-curves/results.json.

Usage:
    cd BackEnd && python scripts/run_all_training.py
    cd BackEnd && python scripts/run_all_training.py --episodes 25
    cd BackEnd && python scripts/run_all_training.py --resume  # skip already-completed configs
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import CampaignResult, ExtensionConfig, result_to_dict
from scripts.run_campaign import train_and_evaluate, save_campaign_results

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
CAMPAIGN_DIR = os.path.join(RESULTS_DIR, "campaigns", "training-curves")
RESULTS_FILE = os.path.join(CAMPAIGN_DIR, "results.json")


def load_existing_results():
    """Load previously saved results (for resume support)."""
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", [])
    return []


def save_incremental(all_results):
    """Save results to disk immediately. Called after every config."""
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, default=str, indent=2)
    logger.info("Saved %d results to %s", len(all_results), RESULTS_FILE)


def config_key(cfg):
    """Unique key for a config to check if already completed."""
    return f"{cfg.get('name', '')}_{cfg.get('topology', '')}_{cfg.get('trainer_type', '')}"


def build_all_configs(n_episodes=25, n_eval_runs=5):
    """Build all training configs needed for complete visualizations."""
    configs = []

    # =====================================================================
    # GROUP 1: Baseline training curves (FedRL, MARL, SARL) x (3x3, 5x5)
    # These fill Fig 5 (learning curves)
    # =====================================================================
    for trainer in ["FedRL", "MARL", "SARL"]:
        for topology in ["grid-3x3", "grid-5x5"]:
            configs.append(ExtensionConfig(
                name=f"{trainer.lower()}_{topology}_training",
                trainer_type=trainer,
                topology=topology,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fedprox_mu=0.0,
                alpha=1.0,
                time_of_day=False,
                use_time_encoding=False,
            ))

    # =====================================================================
    # GROUP 2: FedProx ablation on grid-3x3
    # mu=0.0 is covered by GROUP 1 fedrl_grid-3x3_training
    # =====================================================================
    for mu in [0.01, 0.1]:
        configs.append(ExtensionConfig(
            name=f"fedprox_mu{mu}_grid-3x3",
            trainer_type="FedRL",
            topology="grid-3x3",
            n_episodes=n_episodes,
            n_eval_runs=n_eval_runs,
            fedprox_mu=mu,
            alpha=1.0,
            time_of_day=False,
            use_time_encoding=False,
        ))

    # =====================================================================
    # GROUP 3: Time-of-day ablation on grid-3x3
    # fixed_demand (no ToD, no encoding) is covered by GROUP 1
    # =====================================================================
    configs.append(ExtensionConfig(
        name="tod_with_encoding_grid-3x3",
        trainer_type="FedRL",
        topology="grid-3x3",
        n_episodes=n_episodes,
        n_eval_runs=n_eval_runs,
        fedprox_mu=0.0,
        alpha=1.0,
        time_of_day=True,
        use_time_encoding=True,
    ))

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run all training experiments with incremental save.",
    )
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per config")
    parser.add_argument("--eval-runs", type=int, default=5, help="MC eval runs per config")
    parser.add_argument("--resume", action="store_true", help="Skip already-completed configs")
    args = parser.parse_args()

    configs = build_all_configs(n_episodes=args.episodes, n_eval_runs=args.eval_runs)

    # Load existing results for resume support
    existing_results = load_existing_results() if args.resume else []
    completed_keys = set()
    for r in existing_results:
        cfg = r.get("config", {})
        key = config_key(cfg)
        if not r.get("error"):
            completed_keys.add(key)

    all_results = list(existing_results)
    total = len(configs)
    skipped = 0

    logger.info("=" * 70)
    logger.info("TRAINING PLAN: %d configs, %d episodes each, %d eval runs",
                total, args.episodes, args.eval_runs)
    logger.info("Resume mode: %s (%d already completed)", args.resume, len(completed_keys))
    logger.info("=" * 70)

    for i, cfg in enumerate(configs):
        key = f"{cfg.name}_{cfg.topology}_{cfg.trainer_type}"

        if key in completed_keys:
            logger.info("[%d/%d] SKIP (already done): %s", i + 1, total, cfg.name)
            skipped += 1
            continue

        logger.info("=" * 70)
        logger.info("[%d/%d] STARTING: %s (%s on %s)",
                    i + 1, total, cfg.name, cfg.trainer_type, cfg.topology)
        logger.info("  fedprox_mu=%.2f, alpha=%.1f, time_of_day=%s, use_time_encoding=%s",
                    cfg.fedprox_mu, cfg.alpha, cfg.time_of_day, cfg.use_time_encoding)
        logger.info("=" * 70)

        start = time.time()
        result = train_and_evaluate(cfg)
        elapsed = time.time() - start

        # Convert to serializable dict and save immediately
        result_dict = result_to_dict(result)
        all_results.append(result_dict)
        save_incremental(all_results)

        status = "OK" if not result.error else f"FAILED: {result.error}"
        logger.info("[%d/%d] DONE: %s — %s (%.1f min)",
                    i + 1, total, cfg.name, status, elapsed / 60)

    logger.info("=" * 70)
    logger.info("ALL COMPLETE: %d run, %d skipped, %d total results saved",
                total - skipped, skipped, len(all_results))
    logger.info("Results: %s", RESULTS_FILE)
    logger.info("=" * 70)

    # Print summary table
    print("\n" + "=" * 80)
    print(f"{'Config':<40} {'Status':>8} {'Duration':>10}")
    print("-" * 80)
    for r in all_results:
        cfg = r.get("config", {})
        err = r.get("error")
        dur = r.get("duration_seconds", 0)
        status = "FAIL" if err else "OK"
        print(f"{cfg.get('name', '?'):<40} {status:>8} {dur/60:>8.1f} min")
    print("=" * 80)


if __name__ == "__main__":
    main()
