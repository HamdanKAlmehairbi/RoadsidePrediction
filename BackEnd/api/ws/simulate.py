from fastapi import APIRouter, WebSocket, WebSocketDisconnect

from ..jobs import jobs

router = APIRouter()


@router.websocket("/ws/simulate/{job_id}")
async def websocket_simulate(websocket: WebSocket, job_id: str):
    await websocket.accept()

    if job_id not in jobs:
        await websocket.close(code=1008, reason="Job not found")
        return

    queue = jobs[job_id]["frames_queue"]

    try:
        while True:
            frame = await queue.get()
            await websocket.send_json(frame)
            if frame.get("done"):
                break
    except WebSocketDisconnect:
        pass
    finally:
        try:
            await websocket.close()
        except Exception:
            pass
