"""Load SEAL trainer weights into a pure PyTorch actor.

Bypasses Ray entirely: for the live comparison demo we only need forward
passes to pick actions. Weight files follow the RLlib TorchModelV2 naming
convention documented in ``scripts/weights_schema.md``.
"""
from __future__ import annotations

import logging
import pickle
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)

DEFAULT_HIDDEN = (256, 256)
RANKED_OBS_DIM = 14
UNRANKED_OBS_DIM = 10
DEFAULT_ACTION_DIM = 2

# RLlib TorchModelV2 → PPOActor key mapping.
_RLLIB_TO_ACTOR = {
    "_hidden_layers.0._model.0.weight": "trunk.0.weight",
    "_hidden_layers.0._model.0.bias":   "trunk.0.bias",
    "_hidden_layers.1._model.0.weight": "trunk.2.weight",
    "_hidden_layers.1._model.0.bias":   "trunk.2.bias",
    "_logits._model.0.weight":          "head.weight",
    "_logits._model.0.bias":            "head.bias",
}


class PPOActor(nn.Module):
    """Minimal PPO actor — 2-layer MLP + linear head.

    Matches the default RLlib PPO actor network when configured with
    ``hidden=[256, 256]`` and Tanh activations.
    """

    def __init__(self, obs_dim: int, action_dim: int = DEFAULT_ACTION_DIM,
                 hidden: tuple = DEFAULT_HIDDEN) -> None:
        super().__init__()
        self.obs_dim = obs_dim
        layers: list[nn.Module] = []
        prev = obs_dim
        for h in hidden:
            layers += [nn.Linear(prev, h), nn.Tanh()]
            prev = h
        self.trunk = nn.Sequential(*layers)
        self.head = nn.Linear(prev, action_dim)
        self.eval()

    @torch.no_grad()
    def act(self, obs: np.ndarray) -> int:
        """Deterministic greedy action (argmax of logits)."""
        t = torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0)
        logits = self.head(self.trunk(t))
        return int(torch.argmax(logits, dim=-1).item())


@dataclass
class LoadedPolicy:
    """Loaded policy + metadata needed at inference time."""
    multi_agent: bool
    ctde: bool
    mean_field: bool
    obs_dim: int
    action_dim: int
    # For single-agent policies:
    actor: Optional[PPOActor] = None
    # For multi-agent policies: maps agent_id -> PPOActor
    actors: Optional[Dict[str, PPOActor]] = None

    def pick_action(self, agent_id: str, obs: np.ndarray) -> int:
        if self.multi_agent:
            assert self.actors is not None
            # Agent IDs from the weights dict must match SUMO TLS IDs.
            actor = self.actors.get(agent_id)
            if actor is None:
                # Fallback: grab any actor (shared in practice) so we don't crash.
                actor = next(iter(self.actors.values()))
            return actor.act(obs)
        assert self.actor is not None
        return self.actor.act(obs)


def _rllib_to_actor_state(src: Dict[str, Any]) -> Dict[str, torch.Tensor]:
    out = {}
    for rllib_key, actor_key in _RLLIB_TO_ACTOR.items():
        if rllib_key not in src:
            raise KeyError(f"Missing expected key '{rllib_key}' in policy weights")
        tensor = src[rllib_key]
        if not isinstance(tensor, torch.Tensor):
            tensor = torch.as_tensor(tensor)
        out[actor_key] = tensor
    return out


def _build_actor(state_dict_src: Dict[str, Any], obs_dim: int,
                 action_dim: int = DEFAULT_ACTION_DIM) -> PPOActor:
    actor = PPOActor(obs_dim=obs_dim, action_dim=action_dim)
    converted = _rllib_to_actor_state(state_dict_src)
    actor.load_state_dict(converted, strict=True)
    actor.eval()
    return actor


def load_policy(pkl_path: str, trainer_type: str) -> LoadedPolicy:
    """Load a SEAL .pkl file into a LoadedPolicy.

    Parameters
    ----------
    pkl_path : str
        Absolute path to the weight file.
    trainer_type : str
        One of MARL, SARL, FedRL, Gossip, HierFed, FedDistill, MeanField, CTDE.
        Needed to derive obs_dim for MeanField/CTDE.
    """
    trainer_upper = trainer_type.upper()
    # Derive obs_dim per trainer type.
    if trainer_upper == "CTDE":
        obs_dim = 140  # placeholder — CTDE uses local+global concat; refined below
        ctde = True
        mean_field = False
    elif trainer_upper == "MEANFIELD":
        obs_dim = RANKED_OBS_DIM + 1  # +1 for neighbor mean action
        ctde = False
        mean_field = True
    else:
        obs_dim = RANKED_OBS_DIM
        ctde = False
        mean_field = False

    with open(pkl_path, "rb") as fp:
        raw = pickle.load(fp)

    if isinstance(raw, dict) and raw.get("__multi_policy__"):
        # Multi-agent trainers (MARL, Gossip, HierFed, FedDistill, MeanField, CTDE).
        policies_dict = raw.get("policies") or {
            k: v for k, v in raw.items()
            if k not in ("__multi_policy__", "__ctde__", "__mean_field__")
        }
        ctde_flag = bool(raw.get("__ctde__", ctde))
        actors: Dict[str, PPOActor] = {}
        first_shape = None
        for agent_id, sd in policies_dict.items():
            # Infer obs_dim from the first hidden layer's input features.
            first_weight = sd.get("_hidden_layers.0._model.0.weight")
            if first_weight is not None:
                derived_obs_dim = (first_weight.shape[1]
                                   if hasattr(first_weight, "shape") else obs_dim)
                if first_shape is None:
                    first_shape = derived_obs_dim
                actors[str(agent_id)] = _build_actor(sd, obs_dim=derived_obs_dim)
            else:
                actors[str(agent_id)] = _build_actor(sd, obs_dim=obs_dim)
        final_obs_dim = first_shape or obs_dim
        logger.info("Loaded multi-agent policy: %s (%d agents, obs_dim=%d, ctde=%s)",
                    pkl_path, len(actors), final_obs_dim, ctde_flag)
        return LoadedPolicy(
            multi_agent=True, ctde=ctde_flag, mean_field=mean_field,
            obs_dim=final_obs_dim, action_dim=DEFAULT_ACTION_DIM, actors=actors,
        )

    # Single-agent (SARL, FedRL).
    first_weight = raw.get("_hidden_layers.0._model.0.weight") if isinstance(raw, dict) else None
    if first_weight is not None and hasattr(first_weight, "shape"):
        obs_dim = first_weight.shape[1]
    actor = _build_actor(raw, obs_dim=obs_dim)
    logger.info("Loaded single-agent policy: %s (obs_dim=%d)", pkl_path, obs_dim)
    return LoadedPolicy(
        multi_agent=False, ctde=False, mean_field=False,
        obs_dim=obs_dim, action_dim=DEFAULT_ACTION_DIM, actor=actor,
    )
