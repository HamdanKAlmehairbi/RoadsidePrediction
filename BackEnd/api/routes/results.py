from fastapi import APIRouter, HTTPException

from ..jobs import jobs

router = APIRouter()


@router.get("/api/results/{job_id}")
def get_results(job_id: str):
    if job_id not in jobs:
        raise HTTPException(status_code=404, detail=f"Job not found: {job_id}")

    job = jobs[job_id]
    response = {
        "job_id": job_id,
        "status": job["status"],
        "type": job["type"],
    }

    if job["results"]:
        response.update(job["results"])

    return response
