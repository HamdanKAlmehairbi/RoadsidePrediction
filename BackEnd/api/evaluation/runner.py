"""Evaluation runner for SEAL experiments.

Provides run_trial() to execute a single evaluation episode on any topology
with any trainer type (FedRL, MARL, SARL, fixed-time, max-pressure), and
resolve_weights_path() to auto-discover trained or example weights.

Usage:
    from api.evaluation.runner import run_trial, resolve_weights_path, TRAINERS, TOPOLOGIES

    result = run_trial("FedRL", "grid-3x3", seed=42, ranked=True)
    print(result.per_tls_rewards)
"""
import logging
import os
import shutil
import xml.etree.ElementTree as ET
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from seal import TRIPINFO_OUT_FILENAME
from seal.trainer.util import GLOBAL_POLICY_VAR
from seal.trainer.communication import (
    EDGE2TLS_POLICY,
    TLS2EDGE_POLICY,
    EDGE2TLS_ACTION,
    EDGE2TLS_RANK,
    TLS2EDGE_OBS,
    VEH2TLS_COMM,
)
from ..training_runner import (
    get_net_file,
    ensure_ray,
    load_weights,
    TOPOLOGY_MAP,
    TRAINED_WEIGHTS_DIR,
)

logger = logging.getLogger(__name__)

# All supported trainer types
TRAINERS = ["FedRL", "MARL", "SARL", "MeanField", "Gossip", "CTDE", "HierFed", "FedDistill", "fixed-time", "max-pressure"]

# All supported topologies (derived from TOPOLOGY_MAP)
TOPOLOGIES = list(TOPOLOGY_MAP.keys())

# Output directory for persisted tripinfo XML files
_RESULTS_DIR = os.path.normpath(
    os.path.join(os.path.dirname(__file__), "..", "..", "results", "tripinfo")
)


@dataclass
class TrialResult:
    """Result of a single evaluation trial.

    Attributes:
        trainer: One of "FedRL", "MARL", "SARL", "fixed-time", "max-pressure".
        topology: One of "grid-3x3", "grid-5x5", "grid-7x7".
        seed: Random seed used for route generation.
        ranked: Whether ranked observation features were used.
        per_tls_rewards: Mapping tls_id -> list of per-step rewards.
        tripinfo_path: Absolute path to the copied tripinfo XML for this trial.
        comm_costs: Mapping comm-type string -> total message count (RL only).
        total_steps: Number of simulation steps executed.
        n_vehicles_total: Number of unique vehicles observed during the episode.
    """
    trainer: str
    topology: str
    seed: int
    ranked: bool
    per_tls_rewards: Dict[str, List[float]] = field(default_factory=dict)
    tripinfo_path: str = ""
    comm_costs: Dict[str, int] = field(default_factory=dict)
    total_steps: int = 0
    n_vehicles_total: int = 0


def resolve_weights_path(
    trainer_type: str,
    topology: str,
    ranked: bool,
    vplph: Optional[int] = None,
    training_seed: Optional[int] = None,
) -> Optional[str]:
    """Auto-discover weights for an RL trainer.

    Search order (first match wins):
    1. trained_weights/weights/{trainer}/{topology}/d{vplph}_s{seed}/{ranked}.pkl  (exact)
    2. trained_weights/weights/{trainer}/{topology}/d{vplph}_s*/{ranked}.pkl       (any seed)
    3. trained_weights/weights/{trainer}/{topology}/{ranked}.pkl                   (legacy flat)
    4. example_weights/ICCPS/Final/{trainer}/{topology}/{ranked}.pkl

    Args:
        trainer_type: One of "FedRL", "MARL", "SARL", "fixed-time", "max-pressure".
        topology: Topology name, e.g. "grid-3x3".
        ranked: Whether to look for ranked or unranked weights.
        vplph: Optional demand level used to scope lookups to a specific d{vplph} dir.
        training_seed: Optional training seed used to scope lookups to a specific s{seed} dir.

    Returns:
        Absolute path to the .pkl file, or None for non-RL trainers.

    Raises:
        FileNotFoundError: If no weights file found for an RL trainer.
    """
    if trainer_type in ("fixed-time", "max-pressure"):
        return None

    ranked_str = "ranked" if ranked else "unranked"
    base_dir = os.path.join(TRAINED_WEIGHTS_DIR, "weights", trainer_type, topology)
    candidates = []

    # Candidate 1: exact demand + seed scope
    if vplph is not None and training_seed is not None:
        candidates.append(
            os.path.join(base_dir, f"d{vplph}_s{training_seed}", f"{ranked_str}.pkl")
        )

    # Candidate 2: any seed at this demand
    if vplph is not None:
        import glob
        glob_pattern = os.path.join(base_dir, f"d{vplph}_s*", f"{ranked_str}.pkl")
        matches = sorted(glob.glob(glob_pattern))
        if matches:
            candidates.append(matches[-1])  # most recent seed dir alphabetically

    # Candidate 3: legacy flat layout (pre-demand-scoping)
    candidates.append(os.path.join(base_dir, f"{ranked_str}.pkl"))

    # Candidate 4: example_weights fallback
    backend_dir = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates.append(
        os.path.join(
            backend_dir, "example_weights", "ICCPS", "Final",
            trainer_type, topology, f"{ranked_str}.pkl",
        )
    )

    for path in candidates:
        if os.path.isfile(path):
            logger.info("Using weights: %s", path)
            return path

    raise FileNotFoundError(
        f"No weights found for trainer='{trainer_type}', topology='{topology}', "
        f"ranked={ranked}, vplph={vplph}, training_seed={training_seed}. "
        f"Checked:\n  " + "\n  ".join(candidates)
    )


def run_trial(
    trainer_type: str,
    topology: str,
    seed: int,
    ranked: bool = True,
    weights_path: Optional[str] = None,
    horizon: int = 450,
    alpha: float = 1.0,
    use_time_encoding: bool = False,
    vplph: int = 360,
) -> TrialResult:
    """Execute a single evaluation episode.

    Args:
        trainer_type: One of "FedRL", "MARL", "SARL", "fixed-time", "max-pressure".
        topology: Topology name, e.g. "grid-3x3".
        seed: Seed for deterministic route generation.
        ranked: Whether to use ranked observation features.
        weights_path: Explicit .pkl path for RL trainers. If None, auto-resolved.
        horizon: Maximum episode length in steps.

    Returns:
        TrialResult with per-TLS rewards, tripinfo path, comm costs, step count,
        and unique vehicle count.
    """
    from seal.sumo.env import SumoEnv
    from seal.sumo.mean_field_env import MeanFieldSumoEnv

    if trainer_type not in TRAINERS:
        raise ValueError(
            f"Unknown trainer_type '{trainer_type}'. Must be one of {TRAINERS}"
        )

    is_mean_field = trainer_type == "MeanField"
    is_ctde = trainer_type == "CTDE"
    is_rl = trainer_type in ("FedRL", "MARL", "SARL", "MeanField", "Gossip", "CTDE", "HierFed", "FedDistill")
    is_max_pressure = trainer_type == "max-pressure"

    # Build environment config
    env_config = {
        "gui": False,
        "net-file": get_net_file(topology),
        "rand_routes_on_reset": True,
        "ranked": ranked,
        "use_dynamic_seed": False,
        "rand_route_args": {"seed": seed, "vehicles_per_lane_per_hour": vplph},
        "horizon": horizon,
        "alpha": alpha,
        "use_time_encoding": use_time_encoding,
    }

    # --- Policy setup for RL trainers ---
    policy = None
    multi_policies = None  # Per-agent policies for MARL specialization
    if is_rl:
        ensure_ray()
        from ray.rllib.algorithms.ppo import PPOConfig, PPOTorchPolicy

        # Resolve weights if not explicitly provided
        if weights_path is None:
            weights_path = resolve_weights_path(trainer_type, topology, ranked)

        # Create env temporarily to get spaces
        # MeanField needs the augmented observation space (+1 dim)
        _EnvCls = MeanFieldSumoEnv if is_mean_field else SumoEnv
        _dummy_env = _EnvCls(config=env_config)
        obs_space = _dummy_env.observation_space
        act_space = _dummy_env.action_space
        _dummy_env.close()

        ppo_config = (
            PPOConfig()
            .api_stack(
                enable_rl_module_and_learner=False,
                enable_env_runner_and_connector_v2=False,
            )
            .framework("torch")
        ).to_dict()

        weights_data = load_weights(weights_path)

        # Check for multi-policy format (MARL/Gossip/HierFed/FedDistill/CTDE/MeanField).
        # Note on FedDistill: FedDistillPPOTorchPolicy only overrides loss() (adds KL
        # regularisation toward consensus logits). At eval time compute_single_action()
        # is called, which never invokes loss(). Loading as PPOTorchPolicy is therefore
        # correct — the network architecture and weights are identical. The 0-trip result
        # observed for FedDistill on grid-5x5 at d600 is a policy convergence failure,
        # NOT a code bug. The eval path here is correct.
        if isinstance(weights_data, dict) and weights_data.get("__multi_policy__"):
            # CTDE policies were trained with augmented obs (local + global).
            # Build policies with the augmented obs space so weights load correctly.
            if weights_data.get("__ctde__"):
                n_agents = len(weights_data["policies"])
                local_dim = obs_space.shape[0]
                global_dim = local_dim * n_agents
                total_dim = local_dim + global_dim
                ctde_low = np.full(total_dim, -np.inf, dtype=np.float32)
                ctde_high = np.full(total_dim, np.inf, dtype=np.float32)
                ctde_low[:local_dim] = obs_space.low
                ctde_high[:local_dim] = obs_space.high
                for i in range(n_agents):
                    start = local_dim + i * local_dim
                    ctde_low[start:start + local_dim] = obs_space.low
                    ctde_high[start:start + local_dim] = obs_space.high
                from gymnasium import spaces as gym_spaces
                ctde_obs_space = gym_spaces.Box(
                    low=ctde_low, high=ctde_high, dtype=np.float32
                )
                policy_obs_space = ctde_obs_space
            else:
                policy_obs_space = obs_space

            multi_policies = {}
            for agent_id, agent_weights in weights_data["policies"].items():
                p = PPOTorchPolicy(policy_obs_space, act_space, ppo_config)
                p.set_weights(agent_weights)
                multi_policies[agent_id] = p
            logger.info("Loaded %d per-agent policies from %s", len(multi_policies), weights_path)
        else:
            # Single policy (SARL, FedRL, or legacy MARL averaged weights)
            policy = PPOTorchPolicy(obs_space, act_space, ppo_config)
            policy.set_weights(weights_data)

    # --- Run episode ---
    # MeanField evaluation needs the augmented env to produce correct observations
    _RunEnvCls = MeanFieldSumoEnv if is_mean_field else SumoEnv
    env = _RunEnvCls(config=env_config)
    obs, info = env.reset(seed=seed)
    agent_ids = list(obs.keys())

    per_tls_rewards: Dict[str, List[float]] = defaultdict(list)
    seen_vehicles: set = set()
    total_steps = 0

    while True:
        # --- Compute actions ---
        if is_rl and multi_policies is not None:
            # MARL/CTDE: use per-agent specialized policies
            actions = {}
            for agent_id in agent_ids:
                agent_obs = obs.get(agent_id)
                if agent_obs is not None:
                    agent_policy = multi_policies.get(agent_id)
                    if agent_policy is None:
                        # Fallback for unknown agent IDs: use first available policy
                        agent_policy = next(iter(multi_policies.values()))
                    if is_ctde:
                        # Pad local obs with zeros for the global state portion.
                        # During training, global state was appended; at eval we
                        # use decentralized execution with zero-padded global.
                        local_dim = agent_obs.shape[0]
                        n_agents = len(agent_ids)
                        padded = np.zeros(
                            local_dim + local_dim * n_agents, dtype=np.float32
                        )
                        padded[:local_dim] = agent_obs
                        action, _, _ = agent_policy.compute_single_action(padded)
                    else:
                        action, _, _ = agent_policy.compute_single_action(agent_obs)
                    actions[agent_id] = int(action)
                else:
                    actions[agent_id] = 0
        elif is_rl and policy is not None:
            actions = {}
            for agent_id in agent_ids:
                agent_obs = obs.get(agent_id)
                if agent_obs is not None:
                    action, _, _ = policy.compute_single_action(agent_obs)
                    actions[agent_id] = int(action)
                else:
                    actions[agent_id] = 0
        elif is_max_pressure:
            try:
                import traci
                from ..baselines.max_pressure import compute_max_pressure_actions
                actions = compute_max_pressure_actions(agent_ids)
            except Exception as exc:
                logger.warning(
                    "Max-pressure action computation failed (%s), falling back to stay", exc
                )
                actions = {agent_id: 0 for agent_id in agent_ids}
        else:
            # Fixed-time: always stay in current phase
            actions = {agent_id: 0 for agent_id in agent_ids}

        obs, reward, terminated, truncated, info_dict = env.step(actions)
        total_steps += 1

        # Accumulate per-TLS rewards
        for tls_id, r in reward.items():
            per_tls_rewards[tls_id].append(float(r))

        # Track unique vehicles via TraCI if available
        try:
            import traci
            for vid in traci.vehicle.getIDList():
                seen_vehicles.add(vid)
        except Exception:
            pass

        is_done = terminated.get("__all__", False) or truncated.get("__all__", False)
        if is_done:
            break

    env.close()

    # --- Parse tripinfo ---
    n_vehicles_from_tripinfo = 0
    dest_path = ""
    try:
        tree = ET.parse(TRIPINFO_OUT_FILENAME)
        root = tree.getroot()
        trips = root.findall("tripinfo")
        n_vehicles_from_tripinfo = len(trips)

        # Persist tripinfo to results directory
        os.makedirs(_RESULTS_DIR, exist_ok=True)
        dest_path = os.path.join(
            _RESULTS_DIR,
            f"{trainer_type}_{topology}_d{vplph}_{seed}.xml",
        )
        shutil.copy2(TRIPINFO_OUT_FILENAME, dest_path)
        logger.info("Tripinfo saved to %s (%d trips)", dest_path, n_vehicles_from_tripinfo)
    except FileNotFoundError:
        logger.warning("Tripinfo file not found: %s", TRIPINFO_OUT_FILENAME)
    except ET.ParseError as exc:
        logger.warning("Failed to parse tripinfo: %s", exc)

    # Combine vehicle count: prefer tripinfo (includes completed trips), fall back to TraCI set
    n_vehicles_total = n_vehicles_from_tripinfo or len(seen_vehicles)

    # --- Comm costs (extracted from env callback data if available) ---
    comm_costs: Dict[str, int] = {}
    if is_rl:
        # The SEAL env records comm events; attempt to read from training_data if accessible
        try:
            if hasattr(env, "training_data"):
                for key in (
                    EDGE2TLS_POLICY,
                    TLS2EDGE_POLICY,
                    EDGE2TLS_ACTION,
                    EDGE2TLS_RANK,
                    TLS2EDGE_OBS,
                    VEH2TLS_COMM,
                ):
                    val = env.training_data.get(key, [])
                    comm_costs[key] = int(sum(val)) if val else 0
        except Exception as exc:
            logger.debug("Could not extract comm costs from env: %s", exc)

    return TrialResult(
        trainer=trainer_type,
        topology=topology,
        seed=seed,
        ranked=ranked,
        per_tls_rewards=dict(per_tls_rewards),
        tripinfo_path=dest_path,
        comm_costs=comm_costs,
        total_steps=total_steps,
        n_vehicles_total=n_vehicles_total,
    )
