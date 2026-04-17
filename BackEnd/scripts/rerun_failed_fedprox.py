"""One-shot script to re-run failed FedProx mu configs and merge with existing results.

The original run_extension_ablation.py run had a bug:
  'FedProxPPOTorchPolicy' object has no attribute '_fedprox_mu'
which has been fixed in seal/trainer/fedprox_policy.py by moving the attribute
initialization BEFORE super().__init__().

This script:
1. Loads the existing results.json (which only has grid-5x5 results due to overwrite)
2. Reconstructs a grid-3x3 fedavg_baseline result by re-using trained weights
   (re-evaluation only, no re-training, since the model was already saved)
3. Re-runs fedprox_mu0.01 and fedprox_mu0.1 on both topologies with the fix
4. Saves a complete merged results.json with all 6 configs

Usage:
    cd BackEnd && python scripts/rerun_failed_fedprox.py
"""
import json
import logging
import os
import sys
import time
import glob

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import CampaignResult, ExtensionConfig, result_to_dict, config_to_dict
from scripts.run_campaign import save_campaign_results, train_and_evaluate
from scripts.run_extension_ablation import build_fedprox_configs, run_ablation

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


_BACKEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
_RESULTS_DIR = os.path.join(_BACKEND_DIR, "results", "campaigns", "fedprox-ablation")


def load_existing_results():
    """Load existing fedprox-ablation results.json."""
    results_path = os.path.join(_RESULTS_DIR, "results.json")
    if not os.path.exists(results_path):
        logger.warning("No existing results.json found")
        return []
    with open(results_path) as f:
        data = json.load(f)
    existing = data.get("results", [])
    logger.info("Loaded %d existing results from %s", len(existing), results_path)
    return existing


def find_fedavg_weights(topology: str) -> str:
    """Find the most recently trained fedavg_baseline weights for a topology."""
    trained_dir = os.path.join(_BACKEND_DIR, "trained_weights")
    # Look for FedRL weights in the trained_weights directory
    pattern = os.path.join(trained_dir, "FedRL", "**", "*.pkl")
    candidates = glob.glob(pattern, recursive=True)
    if not candidates:
        # Try alternate path
        pattern2 = os.path.join(trained_dir, "**", "*.pkl")
        candidates = glob.glob(pattern2, recursive=True)

    if candidates:
        # Sort by modification time, most recent first
        candidates.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        logger.info("Found %d trained weight files, using most recent: %s", len(candidates), candidates[0])
        return candidates[0]
    return None


def main():
    """Re-run failed FedProx mu configs and merge with existing good results."""

    # Load existing results - these include grid-5x5 fedavg_baseline (successful)
    existing_raw = load_existing_results()

    # Separate successful from failed
    good_results_raw = [r for r in existing_raw if not r.get("error")]
    failed_results_raw = [r for r in existing_raw if r.get("error")]

    logger.info("Existing: %d good, %d failed", len(good_results_raw), len(failed_results_raw))

    # For grid-3x3 fedavg_baseline: it ran successfully during the first pass
    # but was overwritten. We need to either re-run it or use the trained weights.
    # Since re-training takes 75 minutes, use the saved weights for evaluation only.
    grid3_fedavg_weights = find_fedavg_weights("grid-3x3")

    all_results_raw = list(good_results_raw)  # Start with existing good results

    # Run grid-3x3 fedavg_baseline (evaluation only using saved weights)
    if grid3_fedavg_weights:
        logger.info("Re-evaluating grid-3x3 fedavg_baseline with existing weights: %s", grid3_fedavg_weights)
        cfg_3x3_baseline = ExtensionConfig(
            name="fedavg_baseline",
            trainer_type="FedRL",
            topology="grid-3x3",
            n_episodes=50,
            n_eval_runs=10,
            fedprox_mu=0.0,
            weights_path=grid3_fedavg_weights,  # Skip training, use saved weights
        )
        result_3x3_baseline = train_and_evaluate(cfg_3x3_baseline)
        all_results_raw.append(result_to_dict(result_3x3_baseline))
        logger.info("grid-3x3 fedavg_baseline: error=%s, duration=%.1fs",
                    result_3x3_baseline.error, result_3x3_baseline.duration_seconds)
    else:
        logger.warning("No trained weights found for grid-3x3 fedavg_baseline — will re-train")
        cfg_3x3_baseline = ExtensionConfig(
            name="fedavg_baseline",
            trainer_type="FedRL",
            topology="grid-3x3",
            n_episodes=50,
            n_eval_runs=10,
            fedprox_mu=0.0,
        )
        result_3x3_baseline = train_and_evaluate(cfg_3x3_baseline)
        all_results_raw.append(result_to_dict(result_3x3_baseline))

    # Run fedprox_mu0.01 and fedprox_mu0.1 on BOTH topologies
    for topology in ["grid-3x3", "grid-5x5"]:
        for mu, name in [(0.01, "fedprox_mu0.01"), (0.1, "fedprox_mu0.1")]:
            logger.info("Running %s on %s ...", name, topology)
            cfg = ExtensionConfig(
                name=name,
                trainer_type="FedRL",
                topology=topology,
                n_episodes=50,
                n_eval_runs=10,
                fedprox_mu=mu,
            )
            result = train_and_evaluate(cfg)
            all_results_raw.append(result_to_dict(result))
            logger.info("%s on %s: error=%s, duration=%.1fs",
                        name, topology, result.error, result.duration_seconds)

    # Save merged results
    output_dir = _RESULTS_DIR
    os.makedirs(output_dir, exist_ok=True)
    results_path = os.path.join(output_dir, "results.json")
    with open(results_path, "w", encoding="utf-8") as fp:
        json.dump({"results": all_results_raw}, fp, default=str, indent=2)
    logger.info("Saved %d merged results to %s", len(all_results_raw), results_path)

    # Print summary
    print(f"\n{'='*60}")
    print(f"FedProx re-run complete: {len(all_results_raw)} total results")
    for r in all_results_raw:
        cfg = r.get("config", {})
        err = r.get("error")
        print(f"  {cfg.get('name','?')} ({cfg.get('topology','?')}): {'OK' if not err else 'FAILED'}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
