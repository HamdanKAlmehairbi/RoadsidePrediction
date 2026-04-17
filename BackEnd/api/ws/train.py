from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..jobs import jobs

router = APIRouter()


@router.websocket("/ws/train/{job_id}")
async def websocket_train(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in jobs:
        await websocket.close(code=1008, reason="Job not found")
        return

    queue = jobs[job_id]["frames_queue"]

    try:
        while True:
            frame = await queue.get()
            await websocket.send_json(frame)
            # Training frames don't have "done" flag -- check job status
            if jobs[job_id]["status"] == "complete" and queue.empty():
                break
    except WebSocketDisconnect:
        pass
    finally:
        await websocket.close()
