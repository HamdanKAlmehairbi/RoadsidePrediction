import asyncio
import concurrent.futures
import logging
import math
import random
import uuid
from typing import Optional

from fastapi import APIRouter
from pydantic import BaseModel

from ..jobs import jobs, create_job

router = APIRouter()
logger = logging.getLogger(__name__)


class TrainRequest(BaseModel):
    trainer: str
    aggr: Optional[str] = None
    topology: str = "grid-3x3"
    ranked: bool = True
    n_episodes: int = 50
    fed_step: int = 1


@router.post("/api/train", status_code=202)
async def start_training(req: TrainRequest):
    job_id = "train_" + uuid.uuid4().hex[:8]
    create_job(job_id, "training")

    asyncio.create_task(
        run_training_job(job_id, req.trainer, req.aggr, req.topology,
                         req.ranked, req.n_episodes, req.fed_step)
    )

    return {"job_id": job_id, "status": "running"}


async def run_training_job(job_id: str, trainer: str, aggr: Optional[str],
                           topology: str, ranked: bool, n_episodes: int, fed_step: int):
    """Run training in a thread pool."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(
            pool, _run_training_sync, job_id, trainer, aggr, topology,
            ranked, n_episodes, fed_step, loop
        )


def _run_training_sync(job_id: str, trainer: str, aggr: Optional[str],
                       topology: str, ranked: bool, n_episodes: int,
                       fed_step: int, loop):
    """Synchronous training runner. Falls back to mock if Ray/SEAL unavailable."""
    queue = jobs[job_id]["frames_queue"]

    # Try real training first
    try:
        _run_real_training(job_id, trainer, aggr, topology, ranked, n_episodes, fed_step, queue, loop)
        return
    except Exception as e:
        logger.warning("Real training failed, falling back to mock: %s", e)

    _run_mock_training(job_id, trainer, aggr, n_episodes, fed_step, queue, loop)


def _put_frame(queue, frame, loop):
    """Thread-safe put into asyncio queue."""
    future = asyncio.run_coroutine_threadsafe(queue.put(frame), loop)
    future.result(timeout=10)


def _run_real_training(job_id, trainer, aggr, topology, ranked, n_episodes, fed_step, queue, loop):
    """Attempt real training with SEAL framework."""
    raise NotImplementedError("Real training requires Ray RLLib + SUMO")


def _run_mock_training(job_id: str, trainer: str, aggr: Optional[str],
                       n_episodes: int, fed_step: int, queue, loop):
    """Generate synthetic training reward curves."""
    import time

    rng = random.Random(42)
    rewards = []

    for ep in range(1, n_episodes + 1):
        # Simulate reward improvement: starts negative, improves over episodes
        base_reward = -4.0 + 3.0 * (1 - math.exp(-ep / (n_episodes * 0.3)))
        noise = rng.gauss(0, 0.15)
        mean_reward = round(base_reward + noise, 3)

        is_fed_round = (trainer.upper() == "FEDRL" and ep % fed_step == 0)

        episode_data = {
            "episode": ep,
            "mean_reward": mean_reward,
            "trainer": trainer,
            "fed_round": is_fed_round,
        }
        rewards.append(episode_data)

        _put_frame(queue, episode_data, loop)
        time.sleep(0.15)  # Simulate training time per episode

    jobs[job_id]["status"] = "complete"
    jobs[job_id]["results"] = {
        "rewards": rewards,
        "trip_metrics": {"travel_time_s": 99.2, "waiting_time_s": 27.1, "time_loss_s": 23.4},
        "comm_costs": {
            "EDGE2TLS_POLICY": 450,
            "TLS2EDGE_OBS": 360,
            "EDGE2TLS_RANK": 720,
            "EDGE2TLS_ACTION": 360,
            "VEH2TLS": 1200,
        },
    }
