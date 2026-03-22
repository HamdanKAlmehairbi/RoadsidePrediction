"""WebSocket endpoint for streaming SEAL evaluation campaign progress.

Streams per-trial progress frames from the asyncio queue until the job
completes or the client disconnects.

Frame types sent to the client:
    {"type": "progress", "completed": int, "total": int, "trainer": str, "topology": str}
    {"type": "complete", "job_id": str, "status": str}

Follows the same pattern as ws/train.py.
"""
from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..jobs import jobs

router = APIRouter()


@router.websocket("/ws/evaluate/{job_id}")
async def websocket_evaluate(websocket: WebSocket, job_id: str):
    """Stream evaluation progress frames for *job_id*.

    Closes with code 1008 if the job_id is unknown.  Closes normally after
    the "complete" frame has been sent and the queue is drained.
    """
    await websocket.accept()

    if job_id not in jobs:
        await websocket.close(code=1008, reason="Job not found")
        return

    queue = jobs[job_id]["frames_queue"]

    try:
        while True:
            frame = await queue.get()
            await websocket.send_json(frame)
            # Stop when job is done and queue is drained
            if jobs[job_id]["status"] in ("complete", "error") and queue.empty():
                break
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
