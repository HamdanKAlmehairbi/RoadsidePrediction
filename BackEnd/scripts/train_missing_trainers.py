"""Train the 5 missing RL trainers and save weights to example_weights/.

Trains Gossip, HierFed, FedDistill, MeanField, CTDE on the specified topologies.
After each training run, copies ranked.pkl to:
    example_weights/ICCPS/Final/{Trainer}/{topology}/ranked.pkl

Resumable: skips any (trainer, topology) pair whose ranked.pkl already exists.

Usage:
    # Sequential — laptop (uses 8 workers to saturate CPU cores)
    cd BackEnd
    python scripts/train_missing_trainers.py --num-workers 8

    # Parallel — HPC with 4 GPUs (4 jobs at once, 1 GPU each)
    python scripts/train_missing_trainers.py --parallel 4 --num-gpus 1

    # Dry-run: show what would be trained without running
    python scripts/train_missing_trainers.py --dry-run
"""
import argparse
import logging
import os
import shutil
import subprocess
import sys
import time
from typing import List, Tuple

from tqdm import tqdm

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

logging.basicConfig(
    level=logging.WARNING,   # suppress Ray/RLlib noise during training
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
# Keep our own messages visible
logging.getLogger(__name__).setLevel(logging.INFO)

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

MISSING_TRAINERS = ["MeanField", "CTDE", "Gossip", "HierFed", "FedDistill"]

EXAMPLE_WEIGHTS_ROOT = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "example_weights", "ICCPS", "Final")
)

DEFAULT_TOPOLOGIES  = ["grid-3x3", "grid-5x5"]
DEFAULT_N_EPISODES  = 50
DEFAULT_NUM_WORKERS = 4
DEFAULT_VPLPH       = 360
DEFAULT_SEED        = 54321


def example_weights_path(trainer: str, topology: str) -> str:
    return os.path.join(EXAMPLE_WEIGHTS_ROOT, trainer, topology, "ranked.pkl")


def already_trained(trainer: str, topology: str) -> bool:
    return os.path.isfile(example_weights_path(trainer, topology))


# ---------------------------------------------------------------------------
# Single training run (runs in-process — used for sequential mode)
# ---------------------------------------------------------------------------

def train_one(trainer_type: str, topology: str, n_episodes: int,
              vplph: int, seed: int, num_workers: int, num_gpus: float,
              outer_bar: "tqdm" = None) -> str:
    """Train one (trainer, topology) pair. Returns path to saved ranked.pkl."""
    from api.training_runner import create_trainer, run_training_loop

    t0 = time.time()
    trainer = create_trainer(
        trainer_type=trainer_type,
        topology=topology,
        ranked=True,
        n_episodes=n_episodes,
        vplph=vplph,
        training_seed=seed,
        num_workers=num_workers,
        num_gpus=num_gpus,
    )

    # Inner bar: episodes within this training run
    ep_bar = tqdm(
        total=n_episodes,
        desc=f"  {trainer_type}/{topology}",
        unit="ep",
        leave=False,
        dynamic_ncols=True,
    )

    def on_episode(episode_num, result):
        reward = result.get("mean_reward", 0)
        ep_bar.set_postfix(reward=f"{reward:.3f}")
        ep_bar.update(1)

    out = run_training_loop(trainer, n_episodes=n_episodes, on_episode_done=on_episode)
    ep_bar.close()
    elapsed = time.time() - t0

    src = out.get("weights_path")
    if not src or not os.path.isfile(src):
        raise RuntimeError(f"No weights file found at: {src}")

    dst = example_weights_path(trainer_type, topology)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    shutil.copy2(src, dst)

    if outer_bar:
        outer_bar.write(f"  [DONE] {trainer_type}/{topology} - {elapsed/60:.1f} min -> {dst}")

    return dst


# ---------------------------------------------------------------------------
# Subprocess runner (used for parallel mode — each job gets its own Ray)
# ---------------------------------------------------------------------------

def _subprocess_args(trainer: str, topology: str, n_episodes: int,
                     vplph: int, seed: int, num_workers: int, num_gpus: float) -> List[str]:
    """Build argv for a single-trainer subprocess call."""
    return [
        sys.executable, __file__,
        "--trainers", trainer,
        "--topologies", topology,
        "--n-episodes", str(n_episodes),
        "--vplph", str(vplph),
        "--seed", str(seed),
        "--num-workers", str(num_workers),
        "--num-gpus", str(num_gpus),
        "--force",           # already checked skip before spawning
        "--no-parallel",     # don't recurse
    ]


def run_parallel(jobs: List[Tuple[str, str]], n_parallel: int,
                 n_episodes: int, vplph: int, seed: int,
                 num_workers: int, num_gpus: float) -> Tuple[list, list]:
    """
    Run up to n_parallel training jobs simultaneously as subprocesses.
    Each subprocess gets CUDA_VISIBLE_DEVICES=<slot_index> to isolate GPUs.

    Returns (trained, failed) lists of (trainer, topology) tuples.
    """
    trained, failed = [], []
    pending = list(jobs)
    active: dict = {}  # proc -> (trainer, topology, gpu_slot, start_time)
    gpu_slots = list(range(n_parallel))  # available GPU indices

    def _launch_next():
        if not pending or not gpu_slots:
            return
        trainer, topology = pending.pop(0)
        gpu_id = gpu_slots.pop(0)
        env = {**os.environ, "CUDA_VISIBLE_DEVICES": str(gpu_id)}
        argv = _subprocess_args(trainer, topology, n_episodes, vplph, seed,
                                num_workers, num_gpus)
        logger.info("[GPU %d] Launching %s / %s", gpu_id, trainer, topology)
        proc = subprocess.Popen(argv, env=env)
        active[proc] = (trainer, topology, gpu_id, time.time())

    # Fill all slots initially
    for _ in range(min(n_parallel, len(pending))):
        _launch_next()

    while active:
        for proc in list(active):
            ret = proc.poll()
            if ret is None:
                continue
            trainer, topology, gpu_id, t0 = active.pop(proc)
            gpu_slots.append(gpu_id)
            elapsed = time.time() - t0
            if ret == 0:
                logger.info("[GPU %d] DONE %s / %s in %.1fs", gpu_id, trainer, topology, elapsed)
                trained.append((trainer, topology))
            else:
                logger.error("[GPU %d] FAILED %s / %s (exit %d)", gpu_id, trainer, topology, ret)
                failed.append((trainer, topology))
            _launch_next()
        time.sleep(2)

    return trained, failed


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Train missing RL trainers and save to example_weights/."
    )
    parser.add_argument("--trainers", nargs="+", default=MISSING_TRAINERS, metavar="T")
    parser.add_argument("--topologies", nargs="+", default=DEFAULT_TOPOLOGIES, metavar="TOP")
    parser.add_argument("--n-episodes", type=int, default=DEFAULT_N_EPISODES, dest="n_episodes")
    parser.add_argument("--vplph", type=int, default=DEFAULT_VPLPH)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS, dest="num_workers",
        help="Ray rollout workers per training job (default 4; try 8 on 10-core laptop)",
    )
    parser.add_argument(
        "--num-gpus", type=float, default=0, dest="num_gpus",
        help="GPUs per training job (e.g. 1 for HPC, 0 for CPU-only)",
    )
    parser.add_argument(
        "--parallel", type=int, default=1, metavar="N",
        help="Run N training jobs simultaneously (HPC: set to number of GPUs)",
    )
    parser.add_argument("--no-parallel", action="store_true", dest="no_parallel",
                        help=argparse.SUPPRESS)  # internal flag for subprocess recursion
    parser.add_argument("--dry-run", action="store_true", dest="dry_run")
    parser.add_argument("--force", action="store_true",
                        help="Retrain even if ranked.pkl already exists")
    args = parser.parse_args()

    total = len(args.trainers) * len(args.topologies)

    logger.info("Training plan : %d trainer(s) × %d topology(ies) = %d runs",
                len(args.trainers), len(args.topologies), total)
    logger.info("Trainers      : %s", args.trainers)
    logger.info("Topologies    : %s", args.topologies)
    logger.info("Episodes      : %d", args.n_episodes)
    logger.info("Workers/job   : %d", args.num_workers)
    logger.info("GPUs/job      : %g", args.num_gpus)
    logger.info("Parallel jobs : %d", args.parallel if not args.no_parallel else 1)
    logger.info("Output root   : %s", EXAMPLE_WEIGHTS_ROOT)
    logger.info("")

    # Determine which jobs to run
    jobs_to_run = []
    skipped = []
    for trainer in args.trainers:
        for topology in args.topologies:
            if not args.force and already_trained(trainer, topology):
                logger.info("SKIP %s / %s — ranked.pkl already exists", trainer, topology)
                skipped.append((trainer, topology))
            else:
                jobs_to_run.append((trainer, topology))

    if args.dry_run:
        for trainer, topology in jobs_to_run:
            logger.info("DRY-RUN: would train %s / %s → %s",
                        trainer, topology, example_weights_path(trainer, topology))
        _print_summary([], skipped, [])
        return

    # Run jobs
    if args.parallel > 1 and not args.no_parallel:
        trained, failed = run_parallel(
            jobs_to_run, args.parallel,
            args.n_episodes, args.vplph, args.seed,
            args.num_workers, args.num_gpus,
        )
    else:
        trained, failed = [], []
        outer_bar = tqdm(
            total=len(jobs_to_run),
            desc="Overall progress",
            unit="run",
            dynamic_ncols=True,
        )
        for trainer, topology in jobs_to_run:
            outer_bar.set_description(f"Training {trainer}/{topology}")
            try:
                train_one(trainer, topology, args.n_episodes, args.vplph,
                          args.seed, args.num_workers, args.num_gpus,
                          outer_bar=outer_bar)
                trained.append((trainer, topology))
            except Exception as exc:
                outer_bar.write(f"  [FAIL] {trainer}/{topology}: {exc}")
                failed.append((trainer, topology))
            finally:
                outer_bar.update(1)
        outer_bar.close()

    _print_summary(trained, skipped, failed)
    if failed:
        sys.exit(1)


def _print_summary(trained, skipped, failed):
    logger.info("")
    logger.info("=" * 60)
    logger.info("TRAINING COMPLETE")
    logger.info("  Trained : %d", len(trained))
    logger.info("  Skipped : %d (already existed)", len(skipped))
    logger.info("  Failed  : %d", len(failed))
    if trained:
        logger.info("")
        logger.info("Weights saved:")
        for t, top in trained:
            logger.info("  %s", example_weights_path(t, top))
    if failed:
        logger.info("")
        logger.info("Failures (re-run to retry):")
        for item in failed:
            if len(item) == 2:
                logger.error("  %s / %s", *item)
    logger.info("=" * 60)


if __name__ == "__main__":
    main()
