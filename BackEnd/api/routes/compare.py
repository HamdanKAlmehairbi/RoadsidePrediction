"""REST endpoint for starting a side-by-side comparison session."""
from __future__ import annotations

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field

from api.inference.dual_runner import SideConfig, start_session, stop_session

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/compare", tags=["compare"])


class CompareRequest(BaseModel):
    topology_a: str = Field(..., examples=["grid-3x3"])
    strategy_a: str = Field(..., examples=["Gossip"])
    topology_b: str = Field(..., examples=["grid-3x3"])
    strategy_b: str = Field(..., examples=["fixed-time"])
    horizon: int = Field(3600, ge=60, le=10000)
    step_hz: float = Field(2.0, ge=0.5, le=10.0)


class CompareResponse(BaseModel):
    session_id: str
    ws_url: str


@router.post("/start", response_model=CompareResponse)
def compare_start(req: CompareRequest) -> CompareResponse:
    """Spin up two simulation subprocesses and return the session id.

    The client then opens ``/ws/compare/{session_id}`` to stream frames.
    """
    try:
        session = start_session(
            SideConfig(topology=req.topology_a, strategy=req.strategy_a),
            SideConfig(topology=req.topology_b, strategy=req.strategy_b),
            horizon=req.horizon,
            step_hz=req.step_hz,
        )
    except Exception as exc:  # noqa: BLE001
        logger.exception("Failed to start comparison")
        raise HTTPException(status_code=500, detail=str(exc)) from exc
    return CompareResponse(session_id=session.session_id,
                           ws_url=f"/ws/compare/{session.session_id}")


@router.post("/stop/{session_id}")
def compare_stop(session_id: str) -> dict:
    stop_session(session_id)
    return {"stopped": session_id}
