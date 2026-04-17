"""Persistent results storage for SEAL evaluation campaigns.

Provides save/load/list/delete operations for evaluation results stored as
JSON files in BackEnd/results/evaluations/.  Files survive server restarts
and can be loaded on demand by job ID.

Usage:
    from api.evaluation.store import save_evaluation, load_evaluation, list_evaluations

    path = save_evaluation("eval_abc12345", results_dict)
    data = load_evaluation("eval_abc12345")   # -> dict or None
    all_evals = list_evaluations()            # -> list of summary dicts
"""
import json
import logging
import os
from typing import Optional

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Storage directory — created at import time so callers never need makedirs
# ---------------------------------------------------------------------------

RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "evaluations")
)
os.makedirs(RESULTS_DIR, exist_ok=True)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def save_evaluation(job_id: str, results: dict) -> str:
    """Write *results* as a JSON file to the evaluations store.

    The file is named ``{job_id}.json`` and placed in :data:`RESULTS_DIR`.
    Non-serializable types (e.g. ``datetime``, dataclasses) are coerced to
    strings via ``default=str`` to avoid runtime errors.

    Args:
        job_id: Unique job identifier (e.g. ``"eval_abc12345"``).
        results: Any JSON-serializable dict (or dict that can be coerced via
            ``default=str``).

    Returns:
        Absolute path of the written file.
    """
    path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, default=str)
    logger.info("Saved evaluation results to %s", path)
    return path


def load_evaluation(job_id: str) -> Optional[dict]:
    """Load a previously saved evaluation result from disk.

    Args:
        job_id: Job identifier used when the evaluation was saved.

    Returns:
        Parsed dict from the JSON file, or ``None`` if the file does not exist.
    """
    path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if not os.path.isfile(path):
        return None
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return data


def list_evaluations() -> list:
    """Return a summary list of all stored evaluations, newest first.

    Only top-level metadata fields are included (``job_id``, ``status``,
    ``trainer``, ``topology``, ``created_at``).  The ``campaign`` array with
    ``individual_results`` is intentionally excluded to keep the response
    small.

    Returns:
        List of dicts sorted by ``created_at`` descending (unknown dates sort
        last).  Each dict has at minimum ``{"job_id": str}``.
    """
    summaries = []
    if not os.path.isdir(RESULTS_DIR):
        return summaries

    for filename in os.listdir(RESULTS_DIR):
        if not filename.endswith(".json"):
            continue
        path = os.path.join(RESULTS_DIR, filename)
        try:
            with open(path, "r", encoding="utf-8") as f:
                data = json.load(f)
            summary = {
                "job_id": data.get("job_id", filename[:-5]),
                "status": data.get("status", "unknown"),
                "created_at": data.get("created_at", ""),
            }
            # Optional top-level fields that may be present
            for optional_key in ("trainers", "topologies", "n_runs"):
                if optional_key in data:
                    summary[optional_key] = data[optional_key]
            summaries.append(summary)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Could not read evaluation file %s: %s", path, exc)

    # Sort by created_at descending; unknowns sort to the end
    summaries.sort(key=lambda s: s.get("created_at") or "", reverse=True)
    return summaries


def delete_evaluation(job_id: str) -> bool:
    """Delete a stored evaluation file.

    Args:
        job_id: Job identifier of the evaluation to delete.

    Returns:
        ``True`` if the file was deleted, ``False`` if it was not found.
    """
    path = os.path.join(RESULTS_DIR, f"{job_id}.json")
    if not os.path.isfile(path):
        return False
    os.remove(path)
    logger.info("Deleted evaluation file: %s", path)
    return True
