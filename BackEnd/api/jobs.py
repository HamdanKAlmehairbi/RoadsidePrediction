import asyncio
from typing import Dict, Any

# In-memory store: job_id -> dict with keys: status, type, frames_queue, results
jobs: Dict[str, Dict[str, Any]] = {}


def create_job(job_id: str, job_type: str):
    jobs[job_id] = {
        "status": "running",
        "type": job_type,
        "frames_queue": asyncio.Queue(maxsize=100),
        "results": None,
    }
