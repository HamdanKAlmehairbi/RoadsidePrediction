"""Run the full experiment matrix from the completion plan.

Phase 1: Multi-demand evaluation (4 demands x 3 RL strategies x 2 topologies)
Phase 4: Aggregation variants (naive, reward-weighted, FedProx, soft-update, clustered, partial)
Phase 3: Cooperative reward shaping (alpha sweep)

Each config saves immediately after completing. Resume-safe.

Usage:
    cd BackEnd && python scripts/run_full_experiments.py
    cd BackEnd && python scripts/run_full_experiments.py --resume
    cd BackEnd && python scripts/run_full_experiments.py --phase 1
    cd BackEnd && python scripts/run_full_experiments.py --topology grid-3x3
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
CAMPAIGN_DIR = os.path.join(RESULTS_DIR, "campaigns", "full-experiments")
RESULTS_FILE = os.path.join(CAMPAIGN_DIR, "results.json")


def load_existing_results():
    if os.path.exists(RESULTS_FILE):
        with open(RESULTS_FILE, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data.get("results", [])
    return []


def save_incremental(all_results):
    os.makedirs(CAMPAIGN_DIR, exist_ok=True)
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        json.dump({"results": all_results}, f, default=str, indent=2)


def config_key(cfg):
    return cfg.get("name", "") if isinstance(cfg, dict) else cfg.name


def build_configs(n_episodes=25, n_eval_runs=10, phase=None, topology=None):
    configs = []

    # =================================================================
    # PHASE 1: Multi-demand evaluation
    # 4 demands x 3 RL strategies x 2 topologies = 24 RL configs
    # =================================================================
    if phase is None or phase == 1:
        demands = [
            ("low-150", 150, False, False),
            ("med-360", 360, False, False),
            ("high-600", 600, False, False),
            ("var-tod", 360, True, True),
        ]
        topologies = ["grid-3x3", "grid-5x5"]
        if topology:
            topologies = [topology]

        for demand_name, vplph, tod, time_enc in demands:
            for trainer in ["FedRL", "MARL", "SARL"]:
                for topo in topologies:
                    configs.append(ExtensionConfig(
                        name=f"p1_{trainer.lower()}_{topo}_{demand_name}",
                        trainer_type=trainer,
                        topology=topo,
                        n_episodes=n_episodes,
                        n_eval_runs=n_eval_runs,
                        fedprox_mu=0.0,
                        alpha=1.0,
                        time_of_day=tod,
                        use_time_encoding=time_enc,
                        vplph=vplph,
                    ))

        # Fixed-time baseline at each demand level (eval only, no training)
        for demand_name, vplph, tod, time_enc in demands:
            if tod:
                continue  # Fixed-time doesn't use ToD
            for topo in topologies:
                configs.append(ExtensionConfig(
                    name=f"p1_fixed-time_{topo}_{demand_name}",
                    trainer_type="fixed-time",
                    topology=topo,
                    n_episodes=0,
                    n_eval_runs=n_eval_runs,
                    vplph=vplph,
                    weights_path="__baseline__",
                ))

    # =================================================================
    # PHASE 4: Aggregation variants (FedRL only, both topologies, 360 VPLPH)
    # =================================================================
    if phase is None or phase == 4:
        topologies = ["grid-3x3", "grid-5x5"]
        if topology:
            topologies = [topology]

        for topo in topologies:
            # Naive FedAvg (equal weight)
            configs.append(ExtensionConfig(
                name=f"p4_naive_fedavg_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                aggr="naive",
            ))

            # FedProx mu=0.01
            configs.append(ExtensionConfig(
                name=f"p4_fedprox_mu001_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fedprox_mu=0.01,
            ))

            # FedProx mu=0.1
            configs.append(ExtensionConfig(
                name=f"p4_fedprox_mu01_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fedprox_mu=0.1,
            ))

            # Soft-Update FedAvg (tau=0.5)
            configs.append(ExtensionConfig(
                name=f"p4_soft_tau05_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fed_tau=0.5,
            ))

            # Soft-Update FedAvg (tau=0.8)
            configs.append(ExtensionConfig(
                name=f"p4_soft_tau08_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fed_tau=0.8,
            ))

            # Clustered FedAvg
            configs.append(ExtensionConfig(
                name=f"p4_clustered_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fed_cluster=True,
            ))

            # Partial Personalization (FedPer)
            configs.append(ExtensionConfig(
                name=f"p4_partial_{topo}",
                trainer_type="FedRL",
                topology=topo,
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fed_partial=True,
            ))

    # =================================================================
    # PHASE 3: Cooperative reward shaping (FedRL, grid-3x3, 360 VPLPH)
    # =================================================================
    if phase is None or phase == 3:
        for alpha in [0.5, 0.1]:
            configs.append(ExtensionConfig(
                name=f"p3_fedrl_alpha{alpha}_grid-3x3",
                trainer_type="FedRL",
                topology="grid-3x3",
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                alpha=alpha,
            ))

    # =================================================================
    # PHASE 4 extra: FedProx under different demands (grid-3x3 only)
    # =================================================================
    if phase is None or phase == 4:
        for demand_name, vplph in [("low-150", 150), ("high-600", 600)]:
            configs.append(ExtensionConfig(
                name=f"p4_fedprox_mu001_grid-3x3_{demand_name}",
                trainer_type="FedRL",
                topology="grid-3x3",
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                fedprox_mu=0.01,
                vplph=vplph,
            ))

    # =================================================================
    # PHASE 2: Real-world topology (Cologne-8)
    # Cologne has 157 lanes — 360 VPLPH would create 56k vehicles/hour.
    # Use 50 VPLPH for moderate congestion on a real-world network.
    # =================================================================
    if phase is None or phase == 2:
        for trainer in ["FedRL", "MARL", "SARL"]:
            configs.append(ExtensionConfig(
                name=f"p2_{trainer.lower()}_cologne-8",
                trainer_type=trainer,
                topology="cologne-8",
                n_episodes=n_episodes,
                n_eval_runs=n_eval_runs,
                vplph=50,
            ))

    # =================================================================
    # PHASE 6: Robustness — demand spike (360 -> 700 mid-episode)
    # Handled by run_robustness_test.py separately
    # =================================================================

    # =================================================================
    # PHASE 3 (plan): DQN Algorithm Independence
    # NOT IMPLEMENTED — DQN requires different RLlib setup.
    # Acknowledged as limitation: "findings established under PPO"
    # =================================================================

    return configs


def main():
    parser = argparse.ArgumentParser(
        description="Run full experiment matrix from completion plan.",
    )
    parser.add_argument("--episodes", type=int, default=25, help="Episodes per config")
    parser.add_argument("--eval-runs", type=int, default=10, help="MC eval runs per config")
    parser.add_argument("--resume", action="store_true", help="Skip completed configs")
    parser.add_argument("--phase", type=int, default=None, help="Run only this phase (1, 3, or 4)")
    parser.add_argument("--topology", type=str, default=None, help="Run only this topology")
    args = parser.parse_args()

    configs = build_configs(
        n_episodes=args.episodes,
        n_eval_runs=args.eval_runs,
        phase=args.phase,
        topology=args.topology,
    )

    # Deduplicate
    seen = set()
    unique = []
    for c in configs:
        if c.name not in seen:
            seen.add(c.name)
            unique.append(c)
    configs = unique

    # Load existing for resume
    existing_results = load_existing_results() if args.resume else []
    completed_keys = set()
    for r in existing_results:
        cfg = r.get("config", {})
        if not r.get("error"):
            completed_keys.add(config_key(cfg))

    all_results = list(existing_results)
    total = len(configs)
    skipped = 0

    logger.info("=" * 70)
    logger.info("FULL EXPERIMENT MATRIX")
    logger.info("Configs: %d | Episodes: %d | Eval runs: %d | Phase: %s | Topology: %s",
                total, args.episodes, args.eval_runs,
                args.phase or "all", args.topology or "all")
    logger.info("Resume: %s (%d already completed)", args.resume, len(completed_keys))
    logger.info("=" * 70)

    for i, cfg in enumerate(configs):
        status = "SKIP" if cfg.name in completed_keys else "TODO"
        extras = []
        if cfg.fedprox_mu > 0:
            extras.append(f"mu={cfg.fedprox_mu}")
        if cfg.alpha < 1.0:
            extras.append(f"alpha={cfg.alpha}")
        if cfg.time_of_day:
            extras.append("ToD")
        if cfg.vplph != 360:
            extras.append(f"{cfg.vplph}vplph")
        if cfg.fed_tau < 1.0:
            extras.append(f"tau={cfg.fed_tau}")
        if cfg.fed_cluster:
            extras.append("clustered")
        if cfg.fed_partial:
            extras.append("partial")
        if cfg.aggr != "pos_reward":
            extras.append(f"aggr={cfg.aggr}")
        ext = " [" + ", ".join(extras) + "]" if extras else ""
        logger.info("  %s %3d. %-45s %6s %10s%s",
                    status, i + 1, cfg.name, cfg.trainer_type, cfg.topology, ext)

    logger.info("=" * 70)

    for i, cfg in enumerate(configs):
        if cfg.name in completed_keys:
            logger.info("[%d/%d] SKIP (done): %s", i + 1, total, cfg.name)
            skipped += 1
            continue

        logger.info("=" * 70)
        logger.info("[%d/%d] STARTING: %s (%s on %s)",
                    i + 1, total, cfg.name, cfg.trainer_type, cfg.topology)
        logger.info("=" * 70)

        start = time.time()
        result = train_and_evaluate(cfg)
        elapsed = time.time() - start

        result_dict = result_to_dict(result)
        all_results.append(result_dict)
        save_incremental(all_results)

        status = "OK" if not result.error else f"FAIL: {result.error}"
        logger.info("[%d/%d] DONE: %s — %s (%.1f min)",
                    i + 1, total, cfg.name, status, elapsed / 60)

    # Summary
    n_ok = sum(1 for r in all_results if not r.get("error"))
    n_fail = sum(1 for r in all_results if r.get("error"))
    total_time = sum(r.get("duration_seconds", 0) for r in all_results) / 3600

    print()
    print("=" * 70)
    print(f"COMPLETE: {n_ok} OK, {n_fail} FAILED, {skipped} SKIPPED")
    print(f"Total compute: {total_time:.1f} hours")
    print(f"Results: {RESULTS_FILE}")
    print("=" * 70)

    print()
    print(f"{'Config':<48} {'Status':>6} {'Time':>10}")
    print("-" * 66)
    for r in all_results:
        cfg = r.get("config", {})
        err = r.get("error")
        dur = r.get("duration_seconds", 0)
        st = "FAIL" if err else "OK"
        print(f"{cfg.get('name', '?'):<48} {st:>6} {dur/60:>8.1f} min")


if __name__ == "__main__":
    main()
