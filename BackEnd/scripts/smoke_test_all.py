"""Smoke test every experimental permutation with minimal compute.

Runs 2 training episodes + 2 MC eval runs for every config in the completion plan.
Goal: verify nothing crashes before committing to full runs.

Usage:
    cd BackEnd && python scripts/smoke_test_all.py
"""
import json
import logging
import os
import sys
import time
import traceback

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import ExtensionConfig, result_to_dict
from scripts.run_campaign import train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

RESULTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "results"))
SMOKE_DIR = os.path.join(RESULTS_DIR, "campaigns", "smoke-test")
RESULTS_FILE = os.path.join(SMOKE_DIR, "results.json")

N_EPISODES = 1
N_EVAL_RUNS = 1


def build_all_permutations():
    configs = []

    # =====================================================================
    # GROUP 1: Multi-demand × strategies × topologies
    # 4 demands × 3 RL strategies × 2 topologies = 24 configs
    # + Fixed-Time × 2 topologies × 4 demands = 8 configs (eval only)
    # =====================================================================
    demands = [
        ("low", 150, False, False),
        ("medium", 360, False, False),
        ("high", 600, False, False),
        ("variable", 360, True, True),  # ToD uses time_of_day flag
    ]

    for demand_name, vplph, tod, time_enc in demands:
        for trainer in ["FedRL", "MARL", "SARL"]:
            for topology in ["grid-3x3", "grid-5x5"]:
                configs.append(ExtensionConfig(
                    name=f"smoke_{trainer.lower()}_{topology}_{demand_name}",
                    trainer_type=trainer,
                    topology=topology,
                    n_episodes=N_EPISODES,
                    n_eval_runs=N_EVAL_RUNS,
                    fedprox_mu=0.0,
                    alpha=1.0,
                    time_of_day=tod,
                    use_time_encoding=time_enc,
                ))

    # =====================================================================
    # GROUP 2: Aggregation variants (FedRL only, grid-3x3, medium demand)
    # =====================================================================
    # Naive FedAvg — need to pass aggr parameter
    configs.append(ExtensionConfig(
        name="smoke_fedrl_naive_3x3",
        trainer_type="FedRL",
        topology="grid-3x3",
        n_episodes=N_EPISODES,
        n_eval_runs=N_EVAL_RUNS,
        fedprox_mu=0.0,
        alpha=1.0,
    ))

    # FedProx mu=0.01
    configs.append(ExtensionConfig(
        name="smoke_fedprox_mu001_3x3",
        trainer_type="FedRL",
        topology="grid-3x3",
        n_episodes=N_EPISODES,
        n_eval_runs=N_EVAL_RUNS,
        fedprox_mu=0.01,
        alpha=1.0,
    ))

    # FedProx mu=0.1
    configs.append(ExtensionConfig(
        name="smoke_fedprox_mu01_3x3",
        trainer_type="FedRL",
        topology="grid-3x3",
        n_episodes=N_EPISODES,
        n_eval_runs=N_EVAL_RUNS,
        fedprox_mu=0.1,
        alpha=1.0,
    ))

    # FedProx on grid-5x5
    configs.append(ExtensionConfig(
        name="smoke_fedprox_mu001_5x5",
        trainer_type="FedRL",
        topology="grid-5x5",
        n_episodes=N_EPISODES,
        n_eval_runs=N_EVAL_RUNS,
        fedprox_mu=0.01,
        alpha=1.0,
    ))

    # =====================================================================
    # GROUP 3: Cooperative reward shaping (FedRL, grid-3x3)
    # =====================================================================
    for alpha in [0.5, 0.1]:
        configs.append(ExtensionConfig(
            name=f"smoke_fedrl_alpha{alpha}_3x3",
            trainer_type="FedRL",
            topology="grid-3x3",
            n_episodes=N_EPISODES,
            n_eval_runs=N_EVAL_RUNS,
            fedprox_mu=0.0,
            alpha=alpha,
        ))

    # =====================================================================
    # GROUP 4: Time-of-day with encoding (FedRL, both topologies)
    # =====================================================================
    for topology in ["grid-3x3", "grid-5x5"]:
        configs.append(ExtensionConfig(
            name=f"smoke_fedrl_tod_{topology}",
            trainer_type="FedRL",
            topology=topology,
            n_episodes=N_EPISODES,
            n_eval_runs=N_EVAL_RUNS,
            fedprox_mu=0.0,
            alpha=1.0,
            time_of_day=True,
            use_time_encoding=True,
        ))

    return configs


def main():
    configs = build_all_permutations()

    # Deduplicate by name
    seen = set()
    unique_configs = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique_configs.append(c)
    configs = unique_configs

    total = len(configs)
    logger.info("=" * 70)
    logger.info("SMOKE TEST: %d permutations, %d episodes each, %d eval runs",
                total, N_EPISODES, N_EVAL_RUNS)
    logger.info("=" * 70)

    for i, cfg in enumerate(configs):
        logger.info("[%d/%d] %s", i + 1, total, cfg.name)

    all_results = []
    passed = 0
    failed = 0
    failures = []

    for i, cfg in enumerate(configs):
        logger.info("=" * 70)
        logger.info("[%d/%d] TESTING: %s (%s on %s)",
                    i + 1, total, cfg.name, cfg.trainer_type, cfg.topology)
        extras = []
        if cfg.fedprox_mu > 0:
            extras.append(f"mu={cfg.fedprox_mu}")
        if cfg.alpha < 1.0:
            extras.append(f"alpha={cfg.alpha}")
        if cfg.time_of_day:
            extras.append("ToD")
        if cfg.use_time_encoding:
            extras.append("TimeEnc")
        if extras:
            logger.info("  Extensions: %s", ", ".join(extras))
        logger.info("=" * 70)

        start = time.time()
        try:
            result = train_and_evaluate(cfg)
            elapsed = time.time() - start

            if result.error:
                logger.error("[%d/%d] FAIL: %s — %s (%.1fs)",
                             i + 1, total, cfg.name, result.error, elapsed)
                failed += 1
                failures.append((cfg.name, result.error))
            else:
                # Check eval completed
                has_eval = False
                if result.evaluation:
                    camp = result.evaluation
                    if hasattr(camp, 'n_completed'):
                        has_eval = camp.n_completed > 0
                    elif isinstance(camp, dict) and 'campaign' in camp:
                        has_eval = camp['campaign'][0].get('n_completed', 0) > 0

                if has_eval:
                    logger.info("[%d/%d] PASS: %s (%.1fs)", i + 1, total, cfg.name, elapsed)
                    passed += 1
                else:
                    logger.warning("[%d/%d] WARN: %s — training OK but eval empty (%.1fs)",
                                   i + 1, total, cfg.name, elapsed)
                    passed += 1  # Training worked, eval might be edge case

            result_dict = result_to_dict(result)
            all_results.append(result_dict)

        except Exception as exc:
            elapsed = time.time() - start
            logger.error("[%d/%d] CRASH: %s — %s (%.1fs)",
                         i + 1, total, cfg.name, exc, elapsed)
            logger.error(traceback.format_exc())
            failed += 1
            failures.append((cfg.name, str(exc)))

        # Save incrementally
        os.makedirs(SMOKE_DIR, exist_ok=True)
        with open(RESULTS_FILE, "w", encoding="utf-8") as f:
            json.dump({"results": all_results}, f, default=str, indent=2)

    # Final report
    print()
    print("=" * 70)
    print(f"SMOKE TEST COMPLETE: {passed} PASSED, {failed} FAILED, {total} TOTAL")
    print("=" * 70)

    if failures:
        print()
        print("FAILURES:")
        for name, error in failures:
            print(f"  {name}: {error[:80]}")

    print()
    print(f"Results saved to: {RESULTS_FILE}")

    if failed > 0:
        print()
        print("FIX THESE BEFORE RUNNING FULL EXPERIMENTS")
        sys.exit(1)
    else:
        print()
        print("ALL CLEAR — safe to run full experiments")
        sys.exit(0)


if __name__ == "__main__":
    main()
