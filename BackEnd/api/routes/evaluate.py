"""REST endpoints for SEAL evaluation campaigns.

Provides:
    POST  /api/evaluate          — start async evaluation campaign, returns job_id
    GET   /api/evaluation/{job_id} — fetch full aggregated results for a job
    GET   /api/evaluations         — list all stored evaluations (summary only)

Progress is streamed via WebSocket at /ws/evaluate/{job_id}.

Follows the same async pattern as routes/train.py:
  asyncio.create_task -> ThreadPoolExecutor -> sync worker.
"""
import asyncio
import concurrent.futures
import logging
import uuid
from datetime import datetime, timezone
from typing import List, Optional

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..jobs import jobs, create_job
from ..evaluation.monte_carlo import run_full_campaign, campaign_to_dict
from ..evaluation.store import save_evaluation, load_evaluation, list_evaluations

router = APIRouter()
logger = logging.getLogger(__name__)

# RL trainer types (non-baseline trainers that have weights and support transfer)
_RL_TRAINERS = {"FedRL", "MARL", "SARL", "MeanField", "Gossip", "CTDE", "HierFed", "FedDistill"}


# ---------------------------------------------------------------------------
# Request model
# ---------------------------------------------------------------------------

class EvaluateRequest(BaseModel):
    """Request body for POST /api/evaluate."""

    trainers: Optional[List[str]] = None
    """List of trainer types to evaluate.  Defaults to all 10 (None -> TRAINERS)."""

    topologies: Optional[List[str]] = None
    """List of topologies to evaluate.  Defaults to all 3 (None -> TOPOLOGIES)."""

    n_runs: int = 10
    """Number of Monte Carlo repetitions per (trainer, topology) configuration."""

    ranked: bool = True
    """Whether to use ranked observation features for RL trainers."""

    horizon: int = 450
    """Maximum episode length in simulation steps."""

    include_transfer: bool = False
    """If True, also run the cross-topology transfer matrix for each RL trainer."""


# ---------------------------------------------------------------------------
# REST endpoints
# ---------------------------------------------------------------------------

@router.post("/api/evaluate", status_code=202)
async def start_evaluation(req: EvaluateRequest):
    """Start an evaluation campaign asynchronously.

    Returns immediately with a job_id.  Poll GET /api/evaluation/{job_id} for
    results or connect to WS /ws/evaluate/{job_id} for live progress.
    """
    job_id = "eval_" + uuid.uuid4().hex[:8]
    create_job(job_id, "evaluation")

    asyncio.create_task(run_evaluation_job(job_id, req))

    return {"job_id": job_id, "status": "running"}


@router.get("/api/evaluation/{job_id}")
async def get_evaluation(job_id: str):
    """Retrieve full aggregated results for a completed evaluation job.

    Checks in-memory jobs first (fast path for recently-completed jobs),
    then falls back to loading from disk (survives server restarts).

    Returns 404 if the job_id is unknown.
    """
    # Fast path: in-memory job with results
    if job_id in jobs and jobs[job_id].get("results") is not None:
        return jobs[job_id]["results"]

    # Slow path: load from disk (handles server-restart scenario)
    data = load_evaluation(job_id)
    if data is not None:
        return data

    raise HTTPException(status_code=404, detail=f"Evaluation '{job_id}' not found")


@router.get("/api/evaluations")
async def list_evaluations_endpoint():
    """List all stored evaluation summaries, newest first.

    Returns lightweight metadata only (job_id, status, created_at, etc.).
    Use GET /api/evaluation/{job_id} to retrieve full results.
    """
    return list_evaluations()


# ---------------------------------------------------------------------------
# Async job runner
# ---------------------------------------------------------------------------

async def run_evaluation_job(job_id: str, req: EvaluateRequest):
    """Run the evaluation campaign in a thread pool (non-blocking)."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(
            pool, _run_evaluation_sync, job_id, req, loop
        )


# ---------------------------------------------------------------------------
# Synchronous evaluation worker
# ---------------------------------------------------------------------------

def _run_evaluation_sync(job_id: str, req: EvaluateRequest, loop):
    """Run the full evaluation campaign synchronously in a worker thread.

    Flow:
    1. Broadcast progress frames via asyncio queue for each MC trial.
    2. Call run_full_campaign() which internally calls run_monte_carlo() for
       each (trainer, topology) pair.
    3. Optionally run cross-topology transfer matrix for RL trainers.
    4. Persist results to disk via save_evaluation().
    5. Put a final "complete" frame to the queue.
    """
    queue = jobs[job_id]["frames_queue"]

    def _put_frame(frame):
        """Thread-safe, non-blocking put into the asyncio queue."""
        async def _put_or_drop():
            if queue.full():
                try:
                    queue.get_nowait()  # Drop oldest frame to prevent backpressure
                except asyncio.QueueEmpty:
                    pass
            await queue.put(frame)

        future = asyncio.run_coroutine_threadsafe(_put_or_drop(), loop)
        try:
            future.result(timeout=10)
        except Exception as exc:  # noqa: BLE001
            logger.debug("Frame put timed out: %s", exc)

    def on_progress(completed: int, total: int, config):
        """Callback forwarded to run_monte_carlo(); streams per-trial updates."""
        frame = {
            "type": "progress",
            "completed": completed,
            "total": total,
            "trainer": config.trainer,
            "topology": config.topology,
        }
        _put_frame(frame)

    try:
        # --- Main campaign ---
        campaign_results = run_full_campaign(
            trainers=req.trainers,
            topologies=req.topologies,
            n_runs=req.n_runs,
            ranked=req.ranked,
            on_progress=on_progress,
        )
        results = campaign_to_dict(campaign_results)

        # --- Optional transfer matrix ---
        transfer_results = {}
        if req.include_transfer:
            from ..evaluation.transfer import build_transfer_matrix, transfer_matrix_to_dict

            effective_trainers = req.trainers or []
            rl_trainers = [t for t in effective_trainers if t in _RL_TRAINERS]
            # If trainers was None (all trainers), default to RL subset
            if req.trainers is None:
                rl_trainers = list(_RL_TRAINERS)

            for trainer in rl_trainers:
                logger.info("Running transfer matrix for trainer=%s", trainer)
                try:
                    t_results = build_transfer_matrix(trainer, ranked=req.ranked)
                    transfer_results[trainer] = transfer_matrix_to_dict(t_results)
                except Exception as exc:  # noqa: BLE001
                    logger.error("Transfer matrix failed for %s: %s", trainer, exc)
                    transfer_results[trainer] = {"error": str(exc)}

        if transfer_results:
            results["transfer"] = transfer_results

        # --- Metadata ---
        results["job_id"] = job_id
        results["status"] = "complete"
        results["created_at"] = datetime.now(timezone.utc).isoformat()
        results["trainers"] = req.trainers
        results["topologies"] = req.topologies
        results["n_runs"] = req.n_runs

        # --- Persist to disk ---
        save_evaluation(job_id, results)

        # --- Update in-memory job ---
        jobs[job_id]["status"] = "complete"
        jobs[job_id]["results"] = results

    except Exception as exc:  # noqa: BLE001
        logger.error("Evaluation job %s failed: %s", job_id, exc, exc_info=True)
        error_results = {
            "job_id": job_id,
            "status": "error",
            "error": str(exc),
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        jobs[job_id]["status"] = "error"
        jobs[job_id]["results"] = error_results
        try:
            save_evaluation(job_id, error_results)
        except Exception:  # noqa: BLE001
            pass
        results = error_results

    # --- Final frame signals WebSocket loop to close ---
    _put_frame({"type": "complete", "job_id": job_id, "status": jobs[job_id]["status"]})
