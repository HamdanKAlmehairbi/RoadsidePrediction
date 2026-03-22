"""Thin wrapper around SEAL trainers for API-friendly lifecycle.

Provides:
- Ray as singleton (never shutdown per request)
- Per-episode result streaming via callback
- Topology-to-netfile mapping
- Weight loading for inference
"""
import logging
import os
import pickle

import ray

logger = logging.getLogger(__name__)

# Auto-detect SUMO_HOME if not set
if not os.environ.get("SUMO_HOME"):
    for candidate in [
        r"C:\Program Files (x86)\Eclipse\Sumo",
        r"C:\Program Files\Eclipse\Sumo",
        os.path.expanduser("~/sumo"),
    ]:
        if os.path.isdir(candidate):
            os.environ["SUMO_HOME"] = candidate
            bin_dir = os.path.join(candidate, "bin")
            if bin_dir not in os.environ.get("PATH", ""):
                os.environ["PATH"] = bin_dir + os.pathsep + os.environ.get("PATH", "")
            logger.info("Auto-detected SUMO_HOME: %s", candidate)
            break

# Topology name -> net-file path (relative to BackEnd/)
CONFIGS_DIR = os.path.join(os.path.dirname(__file__), "..", "configs", "SMARTCOMP")
CONFIGS_DIR = os.path.normpath(CONFIGS_DIR)

TOPOLOGY_MAP = {
    "grid-3x3": os.path.join(CONFIGS_DIR, "grid-3x3.net.xml"),
    "grid-5x5": os.path.join(CONFIGS_DIR, "grid-5x5.net.xml"),
    "grid-7x7": os.path.join(CONFIGS_DIR, "grid-7x7.net.xml"),
}

TRAINED_WEIGHTS_DIR = os.path.join(os.path.dirname(__file__), "..", "trained_weights")
TRAINED_WEIGHTS_DIR = os.path.normpath(TRAINED_WEIGHTS_DIR)


def ensure_ray():
    """Initialize Ray as a singleton."""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)


def get_net_file(topology: str) -> str:
    """Resolve topology name to .net.xml path."""
    if topology not in TOPOLOGY_MAP:
        raise ValueError(f"Unknown topology '{topology}'. "
                         f"Available: {list(TOPOLOGY_MAP.keys())}")
    path = TOPOLOGY_MAP[topology]
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Net file not found: {path}")
    return path


def create_trainer(trainer_type: str, topology: str, ranked: bool,
                   fed_step: int = 1, aggr: str = None, n_episodes: int = 50):
    """Create a SEAL trainer instance.

    Returns the trainer object (not yet setup — call .train() or use
    run_training_loop() for per-episode streaming).
    """
    from seal.trainer.fed_agent import FedPolicyTrainer
    from seal.trainer.multi_agent import MultiPolicyTrainer
    from seal.trainer.single_agent import SinglePolicyTrainer

    net_file = get_net_file(topology)
    common_kwargs = {
        "net_file": net_file,
        "ranked": ranked,
        "num_workers": 0,
        "num_gpus": 0,
        "root_dir": [TRAINED_WEIGHTS_DIR],
    }

    trainer_upper = trainer_type.upper()
    if trainer_upper == "FEDRL":
        weight_fn = aggr or "naive"
        # Map API aggr names to internal weight_fn keys
        aggr_map = {
            "naive": "naive",
            "pos-reward": "pos_reward",
            "pos_reward": "pos_reward",
            "neg-reward": "neg_reward",
            "neg_reward": "neg_reward",
            "traffic": "traffic",
        }
        weight_fn = aggr_map.get(weight_fn, "naive")
        trainer = FedPolicyTrainer(
            fed_step=fed_step,
            weight_fn=weight_fn,
            **common_kwargs,
        )
    elif trainer_upper == "MARL":
        trainer = MultiPolicyTrainer(**common_kwargs)
    elif trainer_upper == "SARL":
        trainer = SinglePolicyTrainer(**common_kwargs)
    else:
        raise ValueError(f"Unknown trainer type '{trainer_type}'. "
                         f"Use 'FedRL', 'MARL', or 'SARL'.")

    return trainer


def run_training_loop(trainer, n_episodes: int, on_episode_done=None):
    """Run training with per-episode callback for streaming.

    Args:
        trainer: A SEAL trainer instance (from create_trainer).
        n_episodes: Number of training episodes.
        on_episode_done: Callback(episode_num, result_dict) called after each episode.
            result_dict has keys: episode, mean_reward, episode_len_mean,
            trainer, fed_round.

    Returns:
        dict with final training data including rewards list and comm costs.
    """
    ensure_ray()

    # Setup policies and Ray trainer
    trainer.policies = trainer.on_policy_setup()
    from seal.trainer.util import GLOBAL_POLICY_VAR
    if GLOBAL_POLICY_VAR in trainer.policies:
        raise ValueError(f"Policy key '{GLOBAL_POLICY_VAR}' is reserved.")
    temp = next(iter(trainer.policies.values()))
    trainer.policies[GLOBAL_POLICY_VAR] = temp
    trainer.on_setup()

    rewards_list = []

    for r in range(n_episodes):
        trainer._round = r
        trainer._result = trainer.ray_trainer.train()
        trainer.on_data_recording_step()
        trainer.on_logging_step()

        # Extract metrics
        env_runners = trainer._result.get("env_runners", trainer._result)
        mean_reward = env_runners.get("episode_reward_mean", 0.0)
        ep_len = env_runners.get("episode_len_mean", 0.0)

        is_fed_round = False
        if hasattr(trainer, 'fed_step'):
            if trainer.fed_step is None:
                is_fed_round = True
            elif (r + 1) % trainer.fed_step == 0:
                is_fed_round = True

        episode_data = {
            "episode": r + 1,
            "mean_reward": round(float(mean_reward), 4),
            "episode_len_mean": round(float(ep_len), 2),
            "trainer": trainer.trainer_name,
            "fed_round": is_fed_round,
        }
        rewards_list.append(episode_data)

        if on_episode_done:
            on_episode_done(r + 1, episode_data)

        # Save checkpoint periodically
        if r % trainer.checkpoint_freq == 0:
            trainer.ray_trainer.save(trainer.model_path)

    # Save final policy
    weights = trainer.save_test_policy()
    trainer.ray_trainer.get_policy(GLOBAL_POLICY_VAR).set_weights(weights)

    # Get weights file path
    ranked_str = "ranked" if trainer.ranked else "unranked"
    weights_path = os.path.join(trainer.out_weights_dir, f"{ranked_str}.pkl")

    # Tear down (save + stop ray trainer, but don't shutdown ray)
    trainer.ray_trainer.save(trainer.model_path)
    trainer.ray_trainer.stop()

    return {
        "rewards": rewards_list,
        "weights_path": weights_path,
        "training_data": dict(trainer.training_data),
    }


def build_inference_algorithm(topology: str, ranked: bool):
    """Build a Ray PPO algorithm configured for inference (no training).

    Returns (algorithm, agent_ids) where agent_ids are the TLS IDs.
    """
    from ray.rllib.algorithms.ppo import PPO, PPOConfig, PPOTorchPolicy
    from seal.sumo.env import SumoEnv
    from seal.trainer.util import GLOBAL_POLICY_VAR

    ensure_ray()
    net_file = get_net_file(topology)

    env_config = {
        "gui": False,
        "net-file": net_file,
        "rand_routes_on_reset": True,
        "ranked": ranked,
        "use_dynamic_seed": True,
    }

    # Create a dummy env to get observation/action spaces and agent IDs
    dummy_env = SumoEnv(config=env_config)
    obs_space = dummy_env.observation_space
    act_space = dummy_env.action_space
    agent_ids = list(dummy_env._observe().keys())
    dummy_env.close()

    # Build policies dict: one per agent + global eval policy
    policies = {
        agent_id: (PPOTorchPolicy, obs_space, act_space, {})
        for agent_id in agent_ids
    }
    policies[GLOBAL_POLICY_VAR] = (PPOTorchPolicy, obs_space, act_space, {})

    config = (
        PPOConfig()
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .environment(env=SumoEnv, env_config=env_config)
        .framework("torch")
        .debugging(log_level="ERROR")
        .training(lr=0.001, gamma=0.95)
        .multi_agent(
            policies=policies,
            policy_mapping_fn=lambda agent_id, *args, **kwargs: agent_id,
        )
        .resources(num_gpus=0)
        .env_runners(num_env_runners=0)
    )

    algorithm = config.build()
    return algorithm, agent_ids


def load_weights(weight_filepath: str):
    """Load policy weights from a .pkl file."""
    with open(weight_filepath, "rb") as f:
        return pickle.load(f)
