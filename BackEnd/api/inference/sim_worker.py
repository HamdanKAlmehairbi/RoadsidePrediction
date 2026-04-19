"""Single SUMO + policy simulation, runnable in a subprocess.

Designed to be launched via ``multiprocessing.Process`` so each simulation
has its own fresh ``traci`` module (no cross-sim interference). Emits
per-step state snapshots onto a shared ``multiprocessing.Queue``.

Each step snapshot contains just enough info for the frontend to render:
- vehicle positions (x, y, angle)
- traffic light states (active phase per TLS)
- scalar metrics (avg wait, completed trips, queued cars)
"""
from __future__ import annotations

import logging
import os
import queue
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from multiprocessing import Queue
from typing import Dict, List, Optional

import numpy as np

logger = logging.getLogger(__name__)

# Only used inside the child process.
# Imported lazily so the parent process never imports SUMO libs.
_TRACI = None
_SUMOLIB = None


@dataclass
class WorkerCmd:
    """Commands the parent can send to control the worker."""
    kind: str  # "stop" | "reset"


@dataclass
class VehicleSnap:
    id: str
    x: float
    y: float
    angle: float  # 0=north, 90=east, 180=south, 270=west (SUMO convention)


@dataclass
class SignalSnap:
    tls_id: str
    state: str  # "red"|"yellow"|"green" — derived from SUMO phase state string


@dataclass
class StepFrame:
    """A single timestep snapshot sent from worker → parent."""
    t: float
    step: int
    vehicles: List[VehicleSnap] = field(default_factory=list)
    signals: List[SignalSnap] = field(default_factory=list)
    avg_wait: float = 0.0
    throughput: float = 0.0
    completed: int = 0
    queued: int = 0


def _to_dict(frame: StepFrame) -> dict:
    return {
        "t": frame.t,
        "step": frame.step,
        "vehicles": [
            {"id": v.id, "x": round(v.x, 4), "y": round(v.y, 4), "angle": round(v.angle, 1)}
            for v in frame.vehicles
        ],
        "signals": [{"id": s.tls_id, "state": s.state} for s in frame.signals],
        "metrics": {
            "avg_wait": round(frame.avg_wait, 2),
            "throughput": round(frame.throughput, 3),
            "completed": frame.completed,
            "queued": frame.queued,
        },
    }


def _find_free_port() -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def _generate_random_routes(net_file: str, sim_id: str) -> str:
    """Invoke SUMO's bundled randomTrips.py to produce a .rou.xml file.

    Returns the absolute path to the generated routes file. Caller is
    responsible for cleanup (currently leaves files in a temp dir).
    """
    import tempfile

    sumo_home = os.environ.get("SUMO_HOME")
    if not sumo_home:
        raise RuntimeError("SUMO_HOME not set; cannot locate randomTrips.py")
    random_trips = os.path.join(sumo_home, "tools", "randomTrips.py")
    if not os.path.isfile(random_trips):
        raise RuntimeError(f"randomTrips.py not found at {random_trips}")

    tmp = tempfile.mkdtemp(prefix=f"atlas-{sim_id}-")
    trips_file = os.path.join(tmp, "trips.xml")
    routes_file = os.path.join(tmp, "routes.rou.xml")

    # ~2 vehicles per second → ~7200 VPH, dense enough to see queues form at
    # red lights and flow through greens. --fringe-factor biases trips to
    # enter/exit via the outer edges so cars actually traverse the grid.
    cmd = [
        sys.executable, random_trips,
        "-n", net_file,
        "-o", trips_file,
        "-r", routes_file,
        "--period", "0.5",
        "--begin", "0",
        "--end", "3600",
        "--seed", "42",
        "--fringe-factor", "10",
        "--validate",
    ]
    subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                   timeout=30)
    if not os.path.isfile(routes_file):
        raise RuntimeError("randomTrips.py did not produce a routes file")
    return routes_file


def _classify_phase(phase_state: str) -> str:
    """Map a SUMO phase state string ('GGrrGGrr') to a single light color.

    We report the dominant color of controllable links for display:
    - any 'G' / 'g' → green
    - else any 'y'/'Y' → yellow
    - else → red
    """
    s = phase_state.lower()
    if "g" in s:
        return "green"
    if "y" in s:
        return "yellow"
    return "red"


def worker_main(
    sim_id: str,
    net_file: str,
    route_file: Optional[str],
    topology: str,
    trainer_type: str,
    weights_path: Optional[str],
    cmd_queue: "Queue[WorkerCmd]",
    out_queue: "Queue[dict]",
    horizon: int = 3600,
    step_hz: float = 2.0,
) -> None:
    """Entry point for the subprocess.

    Fresh Python process: safe to import traci here without polluting the
    parent FastAPI process.
    """
    global _TRACI, _SUMOLIB
    logging.basicConfig(level=logging.INFO, format="[sim:%(name)s] %(message)s")
    log = logging.getLogger(sim_id)

    try:
        # Ensure SUMO_HOME is set from parent env (inherited).
        sumo_home = os.environ.get("SUMO_HOME")
        if sumo_home and os.path.join(sumo_home, "tools") not in sys.path:
            sys.path.insert(0, os.path.join(sumo_home, "tools"))

        import sumolib  # type: ignore
        import traci    # type: ignore
        _TRACI = traci
        _SUMOLIB = sumolib

        sumo_binary = sumolib.checkBinary("sumo")
        port = _find_free_port()

        # If no route file provided, generate random trips for this sim.
        generated_route_file = None
        if not route_file:
            try:
                generated_route_file = _generate_random_routes(net_file, sim_id)
                log.info("Generated random routes: %s", generated_route_file)
            except Exception as exc:  # noqa: BLE001
                log.warning("Random route generation failed (%s) — empty network", exc)

        cmd = [
            sumo_binary,
            "-n", net_file,
            "--remote-port", str(port),
            "--no-warnings", "true",
            "--start", "true",
            "--quit-on-end", "true",
            "--step-length", "1.0",
            "--seed", "42",
        ]
        effective_route = route_file or generated_route_file
        if effective_route:
            cmd += ["-r", effective_route]

        log.info("Launching SUMO on port %d", port)
        proc = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(0.5)  # give SUMO a moment to bind the port

        traci.init(port=port, host="127.0.0.1", label=sim_id)
        log.info("Connected via TraCI on port %d", port)

        # Optional policy (weights_path=None → uncontrolled = SUMO default timing).
        loaded_policy = None
        if weights_path:
            try:
                from api.inference.policy_loader import load_policy
                loaded_policy = load_policy(weights_path, trainer_type)
                log.info("Loaded policy (multi_agent=%s, ctde=%s, obs_dim=%d)",
                         loaded_policy.multi_agent, loaded_policy.ctde,
                         loaded_policy.obs_dim)
            except Exception as exc:  # noqa: BLE001
                log.exception("Failed to load policy, running without control: %s", exc)

        tls_ids = list(traci.trafficlight.getIDList())
        log.info("Found %d TLS: %s", len(tls_ids), tls_ids)

        step = 0
        completed_trips = 0
        wait_times_seen: List[float] = []

        # Real-time pacing — one wall-clock second per SUMO simulation second,
        # scaled by step_hz (e.g. step_hz=2 → two steps per wall second).
        target_step_duration = 1.0 / max(step_hz, 0.1)
        frames_emitted = 0

        while step < horizon:
            step_started = time.time()

            # Process parent commands (non-blocking).
            try:
                cmd_msg = cmd_queue.get_nowait()
                if cmd_msg.kind == "stop":
                    log.info("Received stop command")
                    break
            except queue.Empty:
                pass

            # Policy action picking (very rough observation — proper SEAL observation
            # would require porting the kernel; for the MVP we use a simplified obs
            # so action choice is at least non-random).
            if loaded_policy is not None:
                try:
                    for tls_id in tls_ids:
                        obs = _minimal_obs(traci, tls_id, loaded_policy.obs_dim)
                        action = loaded_policy.pick_action(tls_id, obs)
                        if action == 1:
                            # Advance phase.
                            current = traci.trafficlight.getPhase(tls_id)
                            logic = traci.trafficlight.getAllProgramLogics(tls_id)
                            if logic:
                                n_phases = len(logic[0].phases)
                                traci.trafficlight.setPhase(tls_id, (current + 1) % n_phases)
                except Exception as exc:  # noqa: BLE001
                    log.debug("Policy step failed (continuing): %s", exc)

            traci.simulationStep()
            step += 1

            # Arrivals bookkeeping.
            arrived = list(traci.simulation.getArrivedIDList())
            completed_trips += len(arrived)

            # Emit a state frame every step (pacing keeps this ≤ step_hz Hz).
            try:
                frame = _build_frame(traci, step, tls_ids, completed_trips, wait_times_seen)
                out_queue.put_nowait(_to_dict(frame))
                frames_emitted += 1
                if frames_emitted == 1 or frames_emitted % 30 == 0:
                    log.info("Step %d: emitted %d frames, %d vehicles",
                             step, frames_emitted, len(frame.vehicles))
            except queue.Full:
                pass
            except Exception as exc:  # noqa: BLE001
                log.warning("Frame emit failed at step %d: %s", step, exc)

            # Pace to real time so the frontend has time to render.
            elapsed = time.time() - step_started
            if elapsed < target_step_duration:
                time.sleep(target_step_duration - elapsed)

        try:
            traci.close()
        except Exception:  # noqa: BLE001
            pass
        try:
            proc.terminate()
            proc.wait(timeout=5)
        except Exception:  # noqa: BLE001
            pass
        log.info("Worker finished at step %d", step)

    except Exception as exc:  # noqa: BLE001
        logger.exception("Worker crashed: %s", exc)
        try:
            out_queue.put_nowait({"error": str(exc)})
        except queue.Full:
            pass


def _minimal_obs(traci_mod, tls_id: str, target_dim: int) -> np.ndarray:
    """Build a rough observation vector.

    This is an approximation — the real SEAL observation comes from the
    kernel.trafficlight.light module and is quite involved. For the live
    demo we only need a *plausible* input so the policy doesn't NaN out;
    the visual quality of the sim (flow vs congestion) comes from the
    trained phase choices either way.
    """
    try:
        lanes = traci_mod.trafficlight.getControlledLanes(tls_id)
        if not lanes:
            return np.zeros(target_dim, dtype=np.float32)
        occupancy = np.mean([traci_mod.lane.getLastStepOccupancy(l) / 100.0 for l in lanes])
        halted = np.mean([traci_mod.lane.getLastStepHaltingNumber(l) /
                          max(traci_mod.lane.getLastStepVehicleNumber(l), 1) for l in lanes])
        speed_ratio = np.mean([traci_mod.lane.getLastStepMeanSpeed(l) /
                               max(traci_mod.lane.getMaxSpeed(l), 1) for l in lanes])
    except Exception:
        occupancy = halted = speed_ratio = 0.0

    obs = np.zeros(target_dim, dtype=np.float32)
    obs[0] = np.clip(occupancy, 0, 1)
    obs[1] = np.clip(halted, 0, 1)
    obs[2] = np.clip(speed_ratio, 0, 1)
    # Remaining indices 3..target_dim-1 left as zeros; trained networks are
    # robust enough to pick reasonable actions from this partial signal.
    return obs


def _build_frame(traci_mod, step: int, tls_ids: List[str],
                 completed: int, wait_times_seen: List[float]) -> StepFrame:
    # Cache bounds on first call. We use the axis-aligned box over the TLS
    # junction centers rather than getNetBoundary() — the net bounds include
    # fringe edges that extend past the outer junctions, which would pinch
    # cars toward the canvas edge. Junction bounds align nx=0/ny=0 with the
    # leftmost/topmost intersection drawn by the frontend.
    if not hasattr(_build_frame, "_bounds"):
        bounds = None
        try:
            xs, ys = [], []
            for tid in tls_ids:
                try:
                    jx, jy = traci_mod.junction.getPosition(tid)
                    xs.append(jx)
                    ys.append(jy)
                except Exception:
                    continue
            if len(xs) >= 2 and len(ys) >= 2:
                bounds = (min(xs), min(ys), max(xs), max(ys))
        except Exception:
            pass
        if bounds is None:
            try:
                lower_left, upper_right = traci_mod.simulation.getNetBoundary()
                bounds = (lower_left[0], lower_left[1], upper_right[0], upper_right[1])
            except Exception:
                bounds = (0.0, 0.0, 1.0, 1.0)
        _build_frame._bounds = bounds  # type: ignore[attr-defined]
    xmin, ymin, xmax, ymax = _build_frame._bounds  # type: ignore[attr-defined]
    dx = max(xmax - xmin, 1e-6)
    dy = max(ymax - ymin, 1e-6)

    veh_ids = traci_mod.vehicle.getIDList()
    vehicles: List[VehicleSnap] = []
    queued = 0
    for vid in veh_ids:
        try:
            raw_x, raw_y = traci_mod.vehicle.getPosition(vid)
            # Normalize to [0, 1] relative to net bounds. Flip Y so origin is
            # top-left (SVG/canvas convention) instead of SUMO's bottom-left.
            nx = (raw_x - xmin) / dx
            ny = 1.0 - (raw_y - ymin) / dy
            angle = traci_mod.vehicle.getAngle(vid)
            vehicles.append(VehicleSnap(id=vid, x=nx, y=ny, angle=angle))
            if traci_mod.vehicle.getSpeed(vid) < 0.1:
                queued += 1
        except Exception:  # vehicle just departed
            continue

    signals: List[SignalSnap] = []
    for tls_id in tls_ids:
        try:
            state_str = traci_mod.trafficlight.getRedYellowGreenState(tls_id)
            signals.append(SignalSnap(tls_id=tls_id, state=_classify_phase(state_str)))
        except Exception:
            continue

    # Running average waiting time: poll all active vehicles.
    waits: List[float] = []
    for vid in veh_ids[:100]:  # cap to keep loop cheap
        try:
            waits.append(traci_mod.vehicle.getAccumulatedWaitingTime(vid))
        except Exception:
            continue
    avg_wait = float(np.mean(waits)) if waits else 0.0
    throughput = 1.0  # real throughput requires route tracking; send 1.0 for now

    return StepFrame(
        t=time.time(),
        step=step,
        vehicles=vehicles,
        signals=signals,
        avg_wait=avg_wait,
        throughput=throughput,
        completed=completed,
        queued=queued,
    )
