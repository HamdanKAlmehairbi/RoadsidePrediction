"""Orchestrate two simulation workers for side-by-side comparison.

Each side runs in its own subprocess (clean ``traci`` state). The dual
runner owns both worker processes, a pair of IPC queues, and produces a
merged asynchronous stream of frames consumed by the WebSocket handler.
"""
from __future__ import annotations

import asyncio
import logging
import os
import queue
import uuid
from dataclasses import dataclass, field
from multiprocessing import Process, Queue
from typing import AsyncIterator, Dict, Optional

from api.evaluation.campaign_config import resolve_example_weights
from api.inference.sim_worker import WorkerCmd, worker_main
from api.training_runner import get_net_file

logger = logging.getLogger(__name__)

# Non-RL baselines — no weight file needed.
BASELINE_STRATEGIES = {"fixed-time", "max-pressure"}


@dataclass
class SideConfig:
    topology: str
    strategy: str


@dataclass
class ComparisonSession:
    session_id: str
    config_a: SideConfig
    config_b: SideConfig
    proc_a: Process
    proc_b: Process
    cmd_a: Queue
    cmd_b: Queue
    out_a: Queue
    out_b: Queue
    latest_a: Optional[dict] = None
    latest_b: Optional[dict] = None


# In-memory session registry keyed by session_id.
_SESSIONS: Dict[str, ComparisonSession] = {}


def _resolve_weights(topology: str, strategy: str) -> Optional[str]:
    if strategy in BASELINE_STRATEGIES:
        return None
    path = resolve_example_weights(strategy, topology)
    if path and os.path.isfile(path):
        return path
    logger.warning("No weights found for %s/%s — falling back to default timing",
                   strategy, topology)
    return None


def start_session(config_a: SideConfig, config_b: SideConfig,
                  horizon: int = 3600, step_hz: float = 2.0) -> ComparisonSession:
    """Spawn two worker processes and return the session handle."""
    session_id = uuid.uuid4().hex[:12]
    logger.info("Starting session %s: A=%s/%s  B=%s/%s", session_id,
                config_a.strategy, config_a.topology,
                config_b.strategy, config_b.topology)

    net_a = get_net_file(config_a.topology)
    net_b = get_net_file(config_b.topology)
    weights_a = _resolve_weights(config_a.topology, config_a.strategy)
    weights_b = _resolve_weights(config_b.topology, config_b.strategy)

    cmd_a: Queue = Queue(maxsize=8)
    cmd_b: Queue = Queue(maxsize=8)
    out_a: Queue = Queue(maxsize=128)
    out_b: Queue = Queue(maxsize=128)

    proc_a = Process(
        target=worker_main,
        name=f"sim-A-{session_id}",
        kwargs={
            "sim_id": f"A-{session_id}",
            "net_file": net_a,
            "route_file": None,
            "topology": config_a.topology,
            "trainer_type": config_a.strategy,
            "weights_path": weights_a,
            "cmd_queue": cmd_a,
            "out_queue": out_a,
            "horizon": horizon,
            "step_hz": step_hz,
        },
        daemon=True,
    )
    proc_b = Process(
        target=worker_main,
        name=f"sim-B-{session_id}",
        kwargs={
            "sim_id": f"B-{session_id}",
            "net_file": net_b,
            "route_file": None,
            "topology": config_b.topology,
            "trainer_type": config_b.strategy,
            "weights_path": weights_b,
            "cmd_queue": cmd_b,
            "out_queue": out_b,
            "horizon": horizon,
            "step_hz": step_hz,
        },
        daemon=True,
    )
    proc_a.start()
    proc_b.start()

    session = ComparisonSession(
        session_id=session_id,
        config_a=config_a, config_b=config_b,
        proc_a=proc_a, proc_b=proc_b,
        cmd_a=cmd_a, cmd_b=cmd_b,
        out_a=out_a, out_b=out_b,
    )
    _SESSIONS[session_id] = session
    return session


def stop_session(session_id: str) -> None:
    session = _SESSIONS.pop(session_id, None)
    if session is None:
        return
    logger.info("Stopping session %s", session_id)
    for cmd_q in (session.cmd_a, session.cmd_b):
        try:
            cmd_q.put_nowait(WorkerCmd(kind="stop"))
        except queue.Full:
            pass
    for proc in (session.proc_a, session.proc_b):
        proc.join(timeout=4.0)
        if proc.is_alive():
            proc.terminate()


def get_session(session_id: str) -> Optional[ComparisonSession]:
    return _SESSIONS.get(session_id)


async def stream_frames(session: ComparisonSession) -> AsyncIterator[dict]:
    """Async generator that merges both workers' output queues.

    Yields dicts shaped like::

        { "session_id": "...", "sim_a": {...}, "sim_b": {...} }

    at the rate the slower worker produces.
    """
    loop = asyncio.get_running_loop()

    def _poll(q: Queue) -> Optional[dict]:
        try:
            return q.get_nowait()
        except queue.Empty:
            return None

    drain_grace = 0  # number of extra iterations after workers die (for draining buffered frames)
    try:
        while True:
            # Poll both queues in an executor to avoid blocking the loop.
            a = await loop.run_in_executor(None, _poll, session.out_a)
            b = await loop.run_in_executor(None, _poll, session.out_b)

            if a is not None:
                if "error" in a:
                    logger.error("Side A worker error: %s", a["error"])
                else:
                    session.latest_a = a
            if b is not None:
                if "error" in b:
                    logger.error("Side B worker error: %s", b["error"])
                else:
                    session.latest_b = b

            if session.latest_a is not None or session.latest_b is not None:
                yield {
                    "session_id": session.session_id,
                    "sim_a": session.latest_a,
                    "sim_b": session.latest_b,
                }

            # Workers done + queues drained → exit. Use a short grace period so
            # frames still in the queue get delivered to the client.
            both_dead = not session.proc_a.is_alive() and not session.proc_b.is_alive()
            queues_empty = session.out_a.empty() and session.out_b.empty()
            if both_dead and queues_empty:
                drain_grace += 1
                if drain_grace > 10:  # ~1 second grace
                    logger.info("Workers finished and queues drained for session %s",
                                session.session_id)
                    break

            await asyncio.sleep(0.1)
    finally:
        stop_session(session.session_id)
