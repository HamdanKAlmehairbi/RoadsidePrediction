"""HPC ablation campaign runner.

Trains and evaluates ablation configurations:
- Demand sweep (eval-only, reuses existing weights)
- Fed step frequency sweep (retrain + eval)
- Cooperative alpha sweep (retrain + eval)
- FedProx mu sweep (retrain + eval)

All results saved incrementally to results/campaigns/{ablation_name}/.
Resumable: skips configs whose name already appears in results.json.

Usage:
    cd BackEnd
    python hpc/run_ablation.py --ablation demand
    python hpc/run_ablation.py --ablation fed_step
    python hpc/run_ablation.py --ablation alpha
    python hpc/run_ablation.py --ablation fedprox
    python hpc/run_ablation.py --ablation all
"""
import argparse
import json
import logging
import os
import shutil
import sys
import time
from typing import List, Optional

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from api.evaluation.campaign_config import (
    CampaignResult,
    ExtensionConfig,
    resolve_example_weights,
)
from api.evaluation.monte_carlo import MCConfig, run_monte_carlo
from api.training_runner import create_trainer, run_training_loop
from scripts.run_campaign import save_campaign_results, train_and_evaluate

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

BACKEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
ABLATION_WEIGHTS = os.path.join(BACKEND_DIR, "example_weights", "ICCPS", "Ablation")

GRID_TOPOS = ["grid-3x3", "grid-5x5"]
ALL_TOPOS = ["grid-3x3", "grid-5x5", "grid-7x7", "cologne-8"]
TOP_TRAINERS = ["Gossip", "HierFed", "FedRL"]
ALL_RL_TRAINERS = ["MARL", "MeanField", "CTDE", "Gossip", "HierFed", "FedDistill", "FedRL", "SARL"]
BASELINE_TRAINERS = ["fixed-time", "max-pressure"]

N_EVAL_RUNS = 5
N_EPISODES = 50
NUM_WORKERS = 8
NUM_GPUS = 1


def ablation_weights_path(ablation_name: str, trainer: str, topology: str) -> str:
    return os.path.join(ABLATION_WEIGHTS, ablation_name, trainer, topology, "ranked.pkl")


def completed_config_names(campaign_name: str) -> set:
    results_path = os.path.join(
        BACKEND_DIR, "results", "campaigns", campaign_name, "results.json"
    )
    if not os.path.exists(results_path):
        return set()
    try:
        with open(results_path, "r", encoding="utf-8") as fp:
            existing = json.load(fp).get("results", [])
            return {r.get("config", {}).get("name") for r in existing}
    except (json.JSONDecodeError, KeyError, OSError):
        return set()


def train_ablation_config(
    trainer_type: str, topology: str, ablation_tag: str,
    fed_step: int = 1, alpha: float = 1.0, fedprox_mu: float = 0.0,
    num_workers: int = NUM_WORKERS, num_gpus: float = NUM_GPUS,
) -> str:
    """Train a single ablation config. Returns path to saved weights."""
    dst = ablation_weights_path(ablation_tag, trainer_type, topology)
    if os.path.isfile(dst):
        logger.info("SKIP training %s/%s/%s -- weights exist", ablation_tag, trainer_type, topology)
        return dst

    logger.info("Training %s/%s/%s (fed_step=%d, alpha=%.2f, mu=%.3f)",
                ablation_tag, trainer_type, topology, fed_step, alpha, fedprox_mu)

    trainer = create_trainer(
        trainer_type=trainer_type,
        topology=topology,
        ranked=True,
        n_episodes=N_EPISODES,
        fed_step=fed_step,
        alpha=alpha,
        fedprox_mu=fedprox_mu,
        vplph=360,
        training_seed=54321,
        num_workers=num_workers,
        num_gpus=num_gpus,
    )
    out = run_training_loop(trainer, n_episodes=N_EPISODES)
    src = out.get("weights_path")
    if not src or not os.path.isfile(src):
        raise RuntimeError(f"No weights at: {src}")

    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)
    logger.info("Saved %s/%s/%s -> %s", ablation_tag, trainer_type, topology, dst)
    return dst


def eval_config(
    name: str, trainer_type: str, topology: str,
    weights_path: Optional[str], campaign_name: str,
    vplph: int = 360, n_eval_runs: int = N_EVAL_RUNS,
):
    """Evaluate a single config and save incrementally."""
    done = completed_config_names(campaign_name)
    if name in done:
        logger.info("SKIP eval %s -- already in results", name)
        return

    logger.info("Evaluating %s (%s on %s, vplph=%d, %d seeds)",
                name, trainer_type, topology, vplph, n_eval_runs)

    if trainer_type in ("fixed-time", "max-pressure"):
        start_time = time.time()
        camp_result = CampaignResult(
            config=ExtensionConfig(
                name=name, trainer_type=trainer_type, topology=topology,
                n_eval_runs=n_eval_runs, vplph=vplph,
                weights_path="__baseline__",
            )
        )
        try:
            mc_config = MCConfig(
                trainer=trainer_type, topology=topology,
                n_runs=n_eval_runs, base_seed=42,
                ranked=True, horizon=450,
                weights_path=None, vplph=vplph,
            )
            mc_result = run_monte_carlo(mc_config)
            camp_result.evaluation = mc_result
            logger.info("%s -- %d/%d OK", name, mc_result.n_completed, n_eval_runs)
        except Exception as exc:
            logger.error("%s FAILED: %s", name, exc)
            camp_result.error = str(exc)
        camp_result.duration_seconds = round(time.time() - start_time, 2)
    else:
        cfg = ExtensionConfig(
            name=name, trainer_type=trainer_type, topology=topology,
            n_eval_runs=n_eval_runs, weights_path=weights_path, vplph=vplph,
        )
        camp_result = train_and_evaluate(cfg)
        logger.info("%s -- done (%.1fs)", name, camp_result.duration_seconds)

    try:
        save_campaign_results([camp_result], campaign_name=campaign_name)
    except Exception as exc:
        logger.error("Save failed for %s: %s", name, exc)


# ---------------------------------------------------------------------------
# Ablation generators
# ---------------------------------------------------------------------------

def run_cologne_extended_ablation(args):
    """Retrain all 8 RL trainers on cologne-8 with 200 episodes (4x baseline).

    Tests whether RL can outperform fixed-time/max-pressure with more training.
    The baseline (50 episodes) showed all RL losing to baselines on cologne-8.
    """
    campaign = "ablation_cologne_extended"
    tag = "cologne_200ep"
    n_episodes_extended = 200

    for trainer in ALL_RL_TRAINERS:
        dst = ablation_weights_path(tag, trainer, "cologne-8")
        if os.path.isfile(dst):
            logger.info("SKIP training %s/cologne-8/200ep -- weights exist", trainer)
        else:
            logger.info("Training %s/cologne-8 with %d episodes...", trainer, n_episodes_extended)
            t = create_trainer(
                trainer_type=trainer, topology="cologne-8", ranked=True,
                n_episodes=n_episodes_extended, vplph=360, training_seed=54321,
                num_workers=args.num_workers, num_gpus=args.num_gpus,
            )
            out = run_training_loop(t, n_episodes=n_episodes_extended)
            src = out.get("weights_path")
            if not src or not os.path.isfile(src):
                logger.error("No weights for %s/cologne-8/200ep", trainer)
                continue
            os.makedirs(os.path.dirname(dst), exist_ok=True)
            shutil.copy2(src, dst)
            logger.info("Saved %s/cologne-8/200ep -> %s", trainer, dst)

        name = f"{trainer}_cologne-8_200ep"
        eval_config(name, trainer, "cologne-8", dst, campaign)

    for trainer in BASELINE_TRAINERS:
        name = f"{trainer.replace('-','_')}_cologne-8_200ep"
        eval_config(name, trainer, "cologne-8", None, campaign)

    logger.info("Cologne-8 extended training ablation complete.")


def run_demand_ablation(args):
    """Eval-only: reuse existing weights at different demand levels."""
    campaign = "ablation_demand"
    demands = [150, 600]  # 360 already in baseline

    for vplph in demands:
        for trainer in ALL_RL_TRAINERS:
            for topo in ALL_TOPOS:
                weights = resolve_example_weights(trainer, topo)
                if weights is None:
                    logger.warning("No weights for %s/%s -- skip", trainer, topo)
                    continue
                name = f"{trainer}_{topo}_d{vplph}"
                eval_config(name, trainer, topo, weights, campaign, vplph=vplph)

        for trainer in BASELINE_TRAINERS:
            for topo in ALL_TOPOS:
                name = f"{trainer.replace('-','_')}_{topo}_d{vplph}"
                eval_config(name, trainer, topo, None, campaign, vplph=vplph)

    logger.info("Demand ablation complete.")


def run_fed_step_ablation(args):
    """Retrain FedRL/Gossip/HierFed at fed_step=3,5,10 then evaluate."""
    campaign = "ablation_fed_step"
    steps = [3, 5, 10]  # 1 already in baseline

    for fs in steps:
        tag = f"fed_step_{fs}"
        for trainer in TOP_TRAINERS:
            for topo in GRID_TOPOS:
                weights = train_ablation_config(
                    trainer, topo, tag, fed_step=fs,
                    num_workers=args.num_workers, num_gpus=args.num_gpus,
                )
                name = f"{trainer}_{topo}_fs{fs}"
                eval_config(name, trainer, topo, weights, campaign)

    logger.info("Fed step ablation complete.")


def run_alpha_ablation(args):
    """Retrain top trainers at alpha=0.1,0.3,0.7 then evaluate."""
    campaign = "ablation_alpha"
    alphas = [0.1, 0.3, 0.7]  # 1.0 already in baseline

    for alpha in alphas:
        tag = f"alpha_{alpha}"
        for trainer in TOP_TRAINERS:
            for topo in GRID_TOPOS:
                weights = train_ablation_config(
                    trainer, topo, tag, alpha=alpha,
                    num_workers=args.num_workers, num_gpus=args.num_gpus,
                )
                name = f"{trainer}_{topo}_a{alpha}"
                eval_config(name, trainer, topo, weights, campaign)

    logger.info("Alpha ablation complete.")


def run_fedprox_ablation(args):
    """Retrain FedRL at mu=0.01,0.1,1.0 then evaluate."""
    campaign = "ablation_fedprox"
    mus = [0.01, 0.1, 1.0]  # 0.0 already in baseline

    for mu in mus:
        tag = f"fedprox_mu_{mu}"
        for topo in GRID_TOPOS:
            weights = train_ablation_config(
                "FedRL", topo, tag, fedprox_mu=mu,
                num_workers=args.num_workers, num_gpus=args.num_gpus,
            )
            name = f"FedRL_{topo}_mu{mu}"
            eval_config(name, "FedRL", topo, weights, campaign)

    logger.info("FedProx ablation complete.")


def main():
    parser = argparse.ArgumentParser(description="SEAL ablation campaign runner")
    parser.add_argument("--ablation", required=True,
                        choices=["cologne_extended", "demand", "fed_step", "alpha", "fedprox", "all"])
    parser.add_argument("--num-workers", type=int, default=NUM_WORKERS, dest="num_workers")
    parser.add_argument("--num-gpus", type=float, default=NUM_GPUS, dest="num_gpus")
    args = parser.parse_args()

    ablations = {
        "cologne_extended": run_cologne_extended_ablation,
        "demand": run_demand_ablation,
        "fed_step": run_fed_step_ablation,
        "alpha": run_alpha_ablation,
        "fedprox": run_fedprox_ablation,
    }

    if args.ablation == "all":
        for name, fn in ablations.items():
            logger.info("=" * 60)
            logger.info("Starting ablation: %s", name)
            logger.info("=" * 60)
            fn(args)
    else:
        ablations[args.ablation](args)


if __name__ == "__main__":
    main()
