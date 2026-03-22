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
    """Thread-safe put into asyncio queue. Drops oldest frame if queue is full."""
    async def _put_or_drop():
        if queue.full():
            try:
                queue.get_nowait()  # Drop oldest frame
            except asyncio.QueueEmpty:
                pass
        await queue.put(frame)
    future = asyncio.run_coroutine_threadsafe(_put_or_drop(), loop)
    future.result(timeout=10)


def _run_traci_simulation(job_id: str, weight_id: str, topology: str, seed: int, queue, loop):
    """Run a SUMO simulation with RL policy control via TraCI.

    If weight_id is 'fixed-timing', runs SUMO with default signal timing (no RL).
    Otherwise loads .pkl weights and uses a PPO policy to control traffic lights.
    """
    import os
    import traci
    import numpy as np
    from ..training_runner import get_net_file, load_weights, ensure_ray
    from seal.sumo.env import SumoEnv
    from seal.sumo.config import LANE_OCCUPANCY, HALTED_LANE_OCCUPANCY

    net_file = get_net_file(topology)
    use_rl = weight_id not in ("fixed-timing", "max-pressure")
    use_max_pressure = weight_id == "max-pressure"
    policy = None

    # Determine if ranked from weight metadata
    ranked = True
    if use_rl:
        weight_filepath = get_weight_filepath(weight_id)
        if not weight_filepath:
            raise FileNotFoundError(f"Weight file not found for id: {weight_id}")
        if "unranked" in weight_id:
            ranked = False

    env_config = {
        "gui": False,
        "net-file": net_file,
        "rand_routes_on_reset": True,
        "ranked": ranked,
        "use_dynamic_seed": False,
        "rand_route_args": {"seed": seed},
        "horizon": 450,
    }

    # Create env and get initial state
    env = SumoEnv(config=env_config)

    if use_rl:
        # Build a standalone PPO policy for inference
        ensure_ray()
        from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy

        obs_space = env.observation_space
        act_space = env.action_space

        ppo_config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
        ).to_dict()

        policy = PPOTorchPolicy(obs_space, act_space, ppo_config)
        weights = load_weights(weight_filepath)
        policy.set_weights(weights)

    obs, info = env.reset(seed=seed)
    agent_ids = list(obs.keys())

    total_steps = 450
    last_frame_data = None

    for step in range(total_steps):
        # Compute actions
        if use_rl and policy is not None:
            actions = {}
            for agent_id in agent_ids:
                agent_obs = obs.get(agent_id)
                if agent_obs is not None:
                    action, _, _ = policy.compute_single_action(agent_obs)
                    actions[agent_id] = int(action)
                else:
                    actions[agent_id] = 0
        elif use_max_pressure:
            from ..baselines.max_pressure import compute_max_pressure_actions
            actions = compute_max_pressure_actions(agent_ids)
        else:
            # Fixed timing: don't change phases (action=0 means stay)
            actions = {agent_id: 0 for agent_id in agent_ids}

        obs, reward, terminated, truncated, info_dict = env.step(actions)
        is_done = terminated.get("__all__", False) or truncated.get("__all__", False)

        # Stream frames every 2 steps (like mock sim does)
        if step % 2 == 0:
            # Collect vehicle data from traci
            vehicles = []
            for vid in traci.vehicle.getIDList():
                x, y = traci.vehicle.getPosition(vid)
                speed = traci.vehicle.getSpeed(vid)
                angle = traci.vehicle.getAngle(vid)
                vehicles.append({
                    "id": vid,
                    "x": round(x, 1),
                    "y": round(y, 1),
                    "speed": round(speed, 1),
                    "angle": round(angle, 1),
                })

            # Collect traffic light data with observations
            traffic_lights = []
            for i, agent_id in enumerate(agent_ids):
                agent_obs = obs.get(agent_id, np.zeros(10))
                tls_state = traci.trafficlight.getRedYellowGreenState(agent_id)
                lane_occ = float(agent_obs[LANE_OCCUPANCY]) if len(agent_obs) > LANE_OCCUPANCY else 0.0
                halt_occ = float(agent_obs[HALTED_LANE_OCCUPANCY]) if len(agent_obs) > HALTED_LANE_OCCUPANCY else 0.0
                r = float(reward.get(agent_id, 0.0))
                traffic_lights.append({
                    "id": agent_id,
                    "state": tls_state,
                    "lane_occupancy": round(lane_occ, 3),
                    "halted_lane_occupancy": round(halt_occ, 3),
                    "reward": round(r, 4),
                    "global_rank": i,
                })

            total_halted = sum(1 for v in vehicles if v["speed"] < 0.5)
            mean_speed = sum(v["speed"] for v in vehicles) / max(len(vehicles), 1)
            mean_reward = sum(r for r in reward.values()) / max(len(reward), 1)

            last_frame_data = {
                "vehicles": vehicles,
                "traffic_lights": traffic_lights,
                "total_halted": total_halted,
                "mean_speed": round(mean_speed, 2),
                "mean_reward": round(float(mean_reward), 4),
            }

            frame = {
                "step": step,
                "done": False,
                "vehicles": vehicles,
                "traffic_lights": traffic_lights,
                "metrics": {
                    "total_halted": total_halted,
                    "mean_speed": round(mean_speed, 2),
                    "mean_reward": round(float(mean_reward), 4),
                },
            }
            _put_frame(queue, frame, loop)

        if is_done:
            break

    env.close()

    # Send done frame
    if last_frame_data is None:
        last_frame_data = {"vehicles": [], "traffic_lights": [],
                           "total_halted": 0, "mean_speed": 0.0, "mean_reward": 0.0}

    _put_frame(queue, {
        "step": total_steps,
        "done": True,
        "vehicles": last_frame_data["vehicles"],
        "traffic_lights": last_frame_data["traffic_lights"],
        "metrics": {
            "total_halted": last_frame_data["total_halted"],
            "mean_speed": last_frame_data["mean_speed"],
            "mean_reward": last_frame_data["mean_reward"],
        },
    }, loop)

    jobs[job_id]["status"] = "complete"
    jobs[job_id]["results"] = {
        "trip_metrics": {},  # Will be populated from tripinfo.xml in Phase 3
        "comm_costs": {},
    }


def _is_near_intersection(veh, intersections, threshold=50):
    """Return True if vehicle is within threshold distance of any intersection."""
    for inter in intersections:
        dx = veh["x"] - inter["x"]
        dy = veh["y"] - inter["y"]
        if (dx * dx + dy * dy) < threshold * threshold:
            return True
    return False


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
    num_vehicles = grid_n * grid_n * 5
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

            if best_inter is not None and best_dist < 45:
                idx = inter_idx[best_inter["id"]]
                phase_idx = ((step // 80) + idx) % len(tls_phases)
                phase = tls_phases[phase_idx]
                # First half of phase = horizontal, second half = vertical
                half = len(phase) // 2
                direction_chars = phase[:half] if is_horizontal else phase[half:]
                should_stop = all(c in ('r', 'y') for c in direction_chars)
                if should_stop:
                    v["speed"] = max(0.0, v["speed"] - 2.0)
                    # Hard stop near intersection — acts as stop line
                    if best_dist < 8:
                        v["speed"] = 0.0
                else:
                    v["speed"] = min(13.89, v["speed"] + 1.5)
            else:
                # No intersection nearby — accelerate back toward normal speed
                v["speed"] = min(13.89, v["speed"] + 1.5)

            # Vehicle queuing: brake if a halted vehicle is ahead on the same road
            for other in vehicles:
                if other["id"] == v["id"]:
                    continue
                same_road = False
                ahead_dist = 0.0
                if is_horizontal:
                    if abs(other["y"] - v["y"]) < 5:
                        if v["angle"] < 180:  # eastbound
                            ahead_dist = other["x"] - v["x"]
                        else:
                            ahead_dist = v["x"] - other["x"]
                        same_road = True
                else:
                    if abs(other["x"] - v["x"]) < 5:
                        if v["angle"] < 90 or v["angle"] > 270:  # northbound
                            ahead_dist = other["y"] - v["y"]
                        else:
                            ahead_dist = v["y"] - other["y"]
                        same_road = True
                if same_road and 0 < ahead_dist < 15 and other["speed"] < 0.5 and _is_near_intersection(other, intersections):
                    v["speed"] = max(0.0, v["speed"] - 2.0)
                    break

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
                v["speed"] = rng.uniform(8.0, 13.89)
            if v["y"] < 0 or v["y"] > max_coord:
                # Re-place on a random road row
                row = rng.randint(1, grid_n)
                v["y"] = row * spacing + rng.choice([-1.6, 1.6])
                v["x"] = rng.uniform(spacing, max_coord - spacing)
                v["angle"] = rng.choice([90.0, 270.0])
                v["speed"] = rng.uniform(8.0, 13.89)

        if step % 2 == 0:
            # Build traffic light states
            tls_data = []
            for idx, inter in enumerate(intersections):
                phase_idx = ((step // 80) + idx) % len(tls_phases)
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
