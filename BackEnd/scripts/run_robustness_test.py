"""Phase 6: Robustness testing — demand spike mid-episode.

Tests how each strategy handles a sudden demand surge from 360 to 700 VPLPH
at the midpoint of the episode. Measures which strategy degrades least.

Usage:
    cd BackEnd && python scripts/run_robustness_test.py
    cd BackEnd && python scripts/run_robustness_test.py --resume
"""
import argparse
import json
import logging
import os
import sys
import time

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import ExtensionConfig, result_to_dict
from scripts.run_campaign import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
CAMPAIGN_DIR = os.path.join(RESULTS_DIR, "campaigns", "robustness")
RESULTS_FILE = os.path.join(CAMPAIGN_DIR, "results.json")


def load_existing():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            return json.load(f).get("results", [])
    return []


def save_incremental(results):
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": results}, f, default=str, indent=2)


def build_configs(n_episodes=25, n_eval_runs=10):
    """Build robustness test configs.

    Strategy: train at 360 VPLPH (normal), then evaluate at 700 VPLPH (spike).
    This simulates a model trained under normal conditions being deployed
    during a demand surge. Which strategy's trained policy degrades least?
    """
    configs = []

    for trainer in ["FedRL", "MARL", "SARL"]:
        for topo in ["grid-3x3", "grid-5x5"]:
            # Train at normal demand
            configs.append(ExtensionConfig(
                name=f"robust_{trainer.lower()}_{topo}_train360_eval700",
                trainer_type=trainer,
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                vplph=360,  # Train at normal
            ))

    return configs


def main():
    parser = argparse.ArgumentParser(description="Robustness test: demand spike")
    parser.add_argument("--episodes", type=int, default=25)
    parser.add_argument("--eval-runs", type=int, default=10)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    configs = build_configs(args.episodes, args.eval_runs)

    existing = load_existing() if args.resume else []
    completed = set(r["config"]["name"] for r in existing if not r.get("error"))
    all_results = list(existing)

    total = len(configs)
    logger.info("ROBUSTNESS TEST: %d configs", total)

    for i, cfg in enumerate(configs):
        if cfg.name in completed:
            logger.info("[%d/%d] SKIP: %s", i + 1, total, cfg.name)
            continue

        logger.info("[%d/%d] STARTING: %s", i + 1, total, cfg.name)

        # Step 1: Train at normal demand (360 VPLPH)
        start = time.time()
        result = train_and_evaluate(cfg)

        if result.error:
            logger.error("[%d/%d] FAIL (training): %s", i + 1, total, result.error)
            all_results.append(result_to_dict(result))
            save_incremental(all_results)
            continue

        trained_weights = result.weights_path
        training_rewards = result.training_rewards

        # Step 2: Evaluate the trained weights at spike demand (700 VPLPH)
        from api.evaluation.monte_carlo import MCConfig, run_monte_carlo

        mc_config = MCConfig(
            trainer=cfg.trainer_type,
            topology=cfg.topology,
            n_runs=cfg.n_eval_runs,
            base_seed=cfg.base_seed,
            ranked=cfg.ranked,
            horizon=cfg.horizon,
            weights_path=trained_weights,
            use_time_encoding=cfg.use_time_encoding,
            vplph=700,  # Evaluate at spike demand
        )
        spike_eval = run_monte_carlo(mc_config)

        elapsed = time.time() - start

        # Build combined result with both normal and spike eval
        result_dict = result_to_dict(result)
        result_dict["spike_evaluation"] = {
            "vplph": 700,
            "n_completed": spike_eval.n_completed,
            "n_failed": spike_eval.n_failed,
        }
        # Add spike metrics if available
        if hasattr(spike_eval, 'individual_results') and spike_eval.individual_results:
            result_dict["spike_evaluation"]["individual"] = spike_eval.individual_results

        result_dict["duration_seconds"] = round(elapsed, 2)
        all_results.append(result_dict)
        save_incremental(all_results)

        logger.info("[%d/%d] DONE: %s (%.1f min)", i + 1, total, cfg.name, elapsed / 60)

    print()
    print("=" * 60)
    print(f"ROBUSTNESS TEST COMPLETE: {len(all_results)} results")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 60)


if __name__ == "__main__":
    main()
