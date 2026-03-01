import asyncio
import concurrent.futures
import logging
import math
import random
import uuid
from typing import Optional

from fastapi import APIRouter, BackgroundTasks
from pydantic import BaseModel

from ..jobs import jobs, create_job
from .weights import get_weight_filepath

router = APIRouter()
logger = logging.getLogger(__name__)

SUMO_AVAILABLE = False
try:
    import traci
    SUMO_AVAILABLE = True
except ImportError:
    logger.warning("TraCI not available -- simulation will use mock mode")


class SimulateRequest(BaseModel):
    weight_id: str
    topology: str
    seed: int = 42


@router.post("/api/simulate", status_code=202)
async def start_simulation(req: SimulateRequest):
    job_id = "sim_" + uuid.uuid4().hex[:8]
    create_job(job_id, "simulation")

    asyncio.create_task(run_simulation_job(job_id, req.weight_id, req.topology, req.seed))

    return {"job_id": job_id, "status": "running"}


async def run_simulation_job(job_id: str, weight_id: str, topology: str, seed: int):
    """Run simulation in a thread pool to avoid blocking the event loop."""
    loop = asyncio.get_event_loop()
    with concurrent.futures.ThreadPoolExecutor() as pool:
        await loop.run_in_executor(
            pool, _run_simulation_sync, job_id, weight_id, topology, seed, loop
        )


def _run_simulation_sync(job_id: str, weight_id: str, topology: str, seed: int, loop):
    """Synchronous simulation runner. Uses SUMO/TraCI if available, otherwise mock."""
    queue = jobs[job_id]["frames_queue"]

    if SUMO_AVAILABLE:
        try:
            _run_traci_simulation(job_id, weight_id, topology, seed, queue, loop)
            return
        except Exception as e:
            logger.warning("TraCI simulation failed, falling back to mock: %s", e)

    # Mock mode fallback
    _run_mock_simulation(job_id, topology, seed, queue, loop)


def _put_frame(queue, frame, loop):
    """Thread-safe put into asyncio queue."""
    future = asyncio.run_coroutine_threadsafe(queue.put(frame), loop)
    future.result(timeout=10)


def _run_traci_simulation(job_id: str, weight_id: str, topology: str, seed: int, queue, loop):
    """Run a real SUMO simulation via TraCI."""
    import os
    import traci

    net_file = os.path.join(
        os.path.dirname(__file__), "..", "..", "configs", "SMARTCOMP", f"{topology}.net.xml"
    )
    net_file = os.path.normpath(net_file)

    sumo_cmd = [
        "sumo",
        "-n", net_file,
        "--seed", str(seed),
        "--no-step-log", "true",
        "--time-to-teleport", "-1",
        "--no-warnings", "true",
        "--duration-log.disable", "true",
        "--end", "500",
    ]

    label = f"sim_{job_id}"
    traci.start(sumo_cmd, label=label)
    conn = traci.getConnection(label)

    total_steps = 500
    for step in range(total_steps):
        conn.simulationStep()

        if step % 10 == 0:
            # Collect vehicle data
            vehicles = []
            for vid in conn.vehicle.getIDList():
                x, y = conn.vehicle.getPosition(vid)
                speed = conn.vehicle.getSpeed(vid)
                angle = conn.vehicle.getAngle(vid)
                vehicles.append({"id": vid, "x": x, "y": y, "speed": speed, "angle": angle})

            # Collect traffic light data
            traffic_lights = []
            for tls_id in conn.trafficlight.getIDList():
                state = conn.trafficlight.getRedYellowGreenState(tls_id)
                traffic_lights.append({
                    "id": tls_id,
                    "state": state,
                    "lane_occupancy": 0.0,
                    "halted_lane_occupancy": 0.0,
                    "reward": 0.0,
                    "global_rank": 0,
                })

            # Compute metrics
            total_halted = sum(1 for v in vehicles if v["speed"] < 0.1)
            mean_speed = sum(v["speed"] for v in vehicles) / max(len(vehicles), 1)

            frame = {
                "step": step,
                "done": False,
                "vehicles": vehicles,
                "traffic_lights": traffic_lights,
                "metrics": {
                    "total_halted": total_halted,
                    "mean_speed": round(mean_speed, 2),
                    "mean_reward": 0.0,
                },
            }
            _put_frame(queue, frame, loop)

    conn.close()

    # Send done frame with last known data
    _put_frame(queue, {
        "step": total_steps,
        "done": True,
        "vehicles": vehicles,
        "traffic_lights": traffic_lights,
        "metrics": {
            "total_halted": total_halted,
            "mean_speed": round(mean_speed, 2),
            "mean_reward": 0.0,
        },
    }, loop)
    jobs[job_id]["status"] = "complete"
    jobs[job_id]["results"] = {
        "trip_metrics": {"travel_time_s": 0.0, "waiting_time_s": 0.0, "time_loss_s": 0.0},
        "comm_costs": {},
    }


def _run_mock_simulation(job_id: str, topology: str, seed: int, queue, loop):
    """Generate realistic mock simulation frames."""
    import time

    rng = random.Random(seed)

    # Parse topology to get grid size
    grid_sizes = {"grid-3x3": 3, "grid-5x5": 5, "grid-7x7": 7}
    grid_n = grid_sizes.get(topology, 3)
    spacing = 100.0
    max_coord = (grid_n + 1) * spacing

    # Generate intersection positions using SUMO junction naming convention
    # SUMO grid names: column letter (A,B,C,...) + row number (0,1,2,...)
    intersections = []
    for row in range(grid_n):
        for col in range(grid_n):
            x = (col + 1) * spacing
            y = (row + 1) * spacing
            node_id = f"{chr(ord('A') + col)}{row}"
            intersections.append({"id": node_id, "x": x, "y": y})

    # Generate initial vehicles on road segments
    num_vehicles = grid_n * grid_n * 3
    vehicles = []
    for i in range(num_vehicles):
        # Place vehicles along random road segments
        road_axis = rng.choice(["h", "v"])  # horizontal or vertical
        if road_axis == "h":
            row = rng.randint(1, grid_n)
            x = rng.uniform(0, max_coord)
            y = row * spacing + rng.choice([-1.6, 1.6])
            angle = rng.choice([90.0, 270.0])  # east/west along horizontal road
        else:
            col = rng.randint(1, grid_n)
            x = col * spacing + rng.choice([-1.6, 1.6])
            y = rng.uniform(0, max_coord)
            angle = rng.choice([0.0, 180.0])  # north/south along vertical road
        speed = rng.uniform(5.0, 13.89)
        vehicles.append({"id": f"veh_{i}", "x": x, "y": y, "speed": speed, "angle": angle})

    tls_phases = ["GGrr", "rrGG", "yyrr", "rryy"]
    total_steps = 300

    # Pre-compute a lookup from intersection id to index (for TLS phase lookup)
    inter_idx = {inter["id"]: i for i, inter in enumerate(intersections)}

    for step in range(total_steps):
        # Update vehicle positions
        for v in vehicles:
            # --- Red-light stopping logic ---
            is_horizontal = abs(v["angle"] - 90) < 10 or abs(v["angle"] - 270) < 10
            best_inter = None
            best_dist = float("inf")
            for inter in intersections:
                if is_horizontal:
                    # Must be on same row (within 30% of spacing)
                    if abs(v["y"] - inter["y"]) < spacing * 0.3:
                        if v["angle"] < 180:  # eastbound
                            dist = inter["x"] - v["x"]
                        else:  # westbound
                            dist = v["x"] - inter["x"]
                        if 0 < dist < best_dist:
                            best_inter = inter
                            best_dist = dist
                else:
                    # Vertical — must be on same column
                    if abs(v["x"] - inter["x"]) < spacing * 0.3:
                        if v["angle"] < 90 or v["angle"] > 270:  # northbound
                            dist = inter["y"] - v["y"]
                        else:  # southbound
                            dist = v["y"] - inter["y"]
                        if 0 < dist < best_dist:
                            best_inter = inter
                            best_dist = dist

            if best_inter is not None and best_dist < 20:
                idx = inter_idx[best_inter["id"]]
                phase_idx = ((step // 30) + idx) % len(tls_phases)
                phase = tls_phases[phase_idx]
                # First half of phase = horizontal, second half = vertical
                half = len(phase) // 2
                direction_chars = phase[:half] if is_horizontal else phase[half:]
                is_red = all(c == 'r' for c in direction_chars)
                if is_red:
                    v["speed"] = max(0.0, v["speed"] - 2.0)
                else:
                    v["speed"] = min(13.89, v["speed"] + 0.5)
            else:
                # No intersection nearby — accelerate back toward normal speed
                v["speed"] = min(13.89, v["speed"] + 0.5)

            # Small random perturbation for realism
            v["speed"] = max(0.0, min(13.89, v["speed"] + rng.uniform(-0.3, 0.3)))

            rad = math.radians(v["angle"])
            dx = math.sin(rad) * v["speed"] * 0.1
            dy = math.cos(rad) * v["speed"] * 0.1
            v["x"] += dx
            v["y"] += dy

            # Wrap around — snap back onto a valid road position
            if v["x"] < 0 or v["x"] > max_coord:
                # Re-place on a random road column
                col = rng.randint(1, grid_n)
                v["x"] = col * spacing + rng.choice([-1.6, 1.6])
                v["y"] = rng.uniform(spacing, max_coord - spacing)
                v["angle"] = rng.choice([0.0, 180.0])
            if v["y"] < 0 or v["y"] > max_coord:
                # Re-place on a random road row
                row = rng.randint(1, grid_n)
                v["y"] = row * spacing + rng.choice([-1.6, 1.6])
                v["x"] = rng.uniform(spacing, max_coord - spacing)
                v["angle"] = rng.choice([90.0, 270.0])

        if step % 2 == 0:
            # Build traffic light states
            tls_data = []
            for idx, inter in enumerate(intersections):
                phase_idx = ((step // 30) + idx) % len(tls_phases)
                occ = rng.uniform(0.1, 0.8)
                halt_occ = rng.uniform(0.0, occ)
                reward = -(occ + halt_occ) ** 2
                tls_data.append({
                    "id": inter["id"],
                    "state": tls_phases[phase_idx],
                    "lane_occupancy": round(occ, 2),
                    "halted_lane_occupancy": round(halt_occ, 2),
                    "reward": round(reward, 3),
                    "global_rank": idx,
                })

            total_halted = sum(1 for v in vehicles if v["speed"] < 0.5)
            mean_speed = sum(v["speed"] for v in vehicles) / max(len(vehicles), 1)
            mean_reward = sum(t["reward"] for t in tls_data) / max(len(tls_data), 1)

            frame = {
                "step": step,
                "done": False,
                "vehicles": [
                    {"id": v["id"], "x": round(v["x"], 1), "y": round(v["y"], 1),
                     "speed": round(v["speed"], 1), "angle": round(v["angle"], 1)}
                    for v in vehicles
                ],
                "traffic_lights": tls_data,
                "metrics": {
                    "total_halted": total_halted,
                    "mean_speed": round(mean_speed, 2),
                    "mean_reward": round(mean_reward, 3),
                },
            }
            _put_frame(queue, frame, loop)

        time.sleep(0.016)  # ~60fps pacing for mock

    # Send done frame with last known data
    _put_frame(queue, {
        "step": total_steps,
        "done": True,
        "vehicles": [
            {"id": v["id"], "x": round(v["x"], 1), "y": round(v["y"], 1),
             "speed": round(v["speed"], 1), "angle": round(v["angle"], 1)}
            for v in vehicles
        ],
        "traffic_lights": tls_data,
        "metrics": {
            "total_halted": total_halted,
            "mean_speed": round(mean_speed, 2),
            "mean_reward": round(mean_reward, 3),
        },
    }, loop)
    jobs[job_id]["status"] = "complete"
    jobs[job_id]["results"] = {
        "trip_metrics": {"travel_time_s": 99.2, "waiting_time_s": 27.1, "time_loss_s": 23.4},
        "comm_costs": {
            "EDGE2TLS_POLICY": 450,
            "TLS2EDGE_OBS": 360,
            "EDGE2TLS_RANK": 720,
            "EDGE2TLS_ACTION": 360,
            "VEH2TLS": 1200,
        },
    }
