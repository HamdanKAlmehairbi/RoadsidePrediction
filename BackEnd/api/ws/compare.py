"""WebSocket endpoint that streams merged frames from both simulations."""
from __future__ import annotations

import asyncio
import logging

from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from api.inference.dual_runner import get_session, stop_session, stream_frames

logger = logging.getLogger(__name__)

router = APIRouter(tags=["compare-ws"])


@router.websocket("/ws/compare/{session_id}")
async def compare_ws(websocket: WebSocket, session_id: str) -> None:
    await websocket.accept()
    session = get_session(session_id)
    if session is None:
        await websocket.send_json({"type": "error", "message": "unknown session_id"})
        await websocket.close()
        return

    logger.info("WS connected for session %s", session_id)
    try:
        async for frame in stream_frames(session):
            await websocket.send_json({"type": "frame", **frame})
    except WebSocketDisconnect:
        logger.info("Client disconnected from session %s", session_id)
    except asyncio.CancelledError:
        logger.info("Session %s cancelled", session_id)
    except Exception:  # noqa: BLE001
        logger.exception("WS error for session %s", session_id)
    finally:
        stop_session(session_id)
        try:
            await websocket.close()
        except Exception:  # noqa: BLE001
            pass
