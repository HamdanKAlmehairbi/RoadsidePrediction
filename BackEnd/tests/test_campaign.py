"""Smoke tests for campaign config module (ExtensionConfig, CampaignResult, EXAMPLE_WEIGHTS_MAP).

These tests are pure unit tests — no SUMO, no Ray, no network required.
They validate dataclass defaults, the weights map structure, and serialization helpers.

Run:
    cd BackEnd && python -m pytest tests/test_campaign.py -x -v
"""
import os
import sys

# Ensure BackEnd is on sys.path so imports work when run from repo root or BackEnd/
_BACKEND_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), ".."))
if _BACKEND_DIR not in sys.path:
    sys.path.insert(0, _BACKEND_DIR)

import pytest

from api.evaluation.campaign_config import (
    CampaignResult,
    EXAMPLE_WEIGHTS_MAP,
    ExtensionConfig,
    config_to_dict,
    resolve_example_weights,
    result_to_dict,
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_extension_config_defaults():
    """ExtensionConfig default values match the plan specification."""
    cfg = ExtensionConfig(name="test", trainer_type="FedRL", topology="grid-3x3")

    assert cfg.n_episodes == 50
    assert cfg.n_eval_runs == 10
    assert cfg.base_seed == 42
    assert cfg.alpha == 1.0
    assert cfg.fedprox_mu == 0.0
    assert cfg.ranked is True
    assert cfg.horizon == 450
    assert cfg.time_of_day is False
    assert cfg.use_time_encoding is False
    assert cfg.weights_path is None


def test_example_weights_map_paths_exist():
    """Every path in EXAMPLE_WEIGHTS_MAP resolves to an actual .pkl file on disk."""
    assert len(EXAMPLE_WEIGHTS_MAP) == 6, (
        f"Expected 6 entries in EXAMPLE_WEIGHTS_MAP, got {len(EXAMPLE_WEIGHTS_MAP)}"
    )

    _BACKEND_DIR = os.path.normpath(
        os.path.join(os.path.dirname(__file__), "..")
    )
    for (trainer, topology), relative_path in EXAMPLE_WEIGHTS_MAP.items():
        abs_path = os.path.normpath(os.path.join(_BACKEND_DIR, relative_path))
        assert os.path.isfile(abs_path), (
            f"Missing example weights for ({trainer}, {topology}): {abs_path}"
        )


def test_resolve_example_weights_found():
    """resolve_example_weights returns an absolute path ending in the correct filename."""
    path = resolve_example_weights("FedRL", "grid-3x3")
    assert path is not None
    assert path.endswith("v3_naive-aggr_ranked.pkl"), (
        f"Expected path ending in 'v3_naive-aggr_ranked.pkl', got: {path}"
    )
    assert os.path.isabs(path), f"Expected absolute path, got: {path}"


def test_resolve_example_weights_missing():
    """resolve_example_weights returns None for a topology with no example weights."""
    path = resolve_example_weights("FedRL", "grid-7x7")
    assert path is None, (
        f"Expected None for FedRL/grid-7x7, got: {path}"
    )


def test_config_to_dict_roundtrip():
    """config_to_dict returns a dict with all expected keys and correct values."""
    cfg = ExtensionConfig(
        name="fedprox_mu0.1",
        trainer_type="FedRL",
        topology="grid-5x5",
        n_episodes=30,
        n_eval_runs=5,
        fedprox_mu=0.1,
        alpha=0.5,
    )
    d = config_to_dict(cfg)

    assert isinstance(d, dict)
    expected_keys = {
        "name", "trainer_type", "topology", "n_episodes", "n_eval_runs",
        "base_seed", "ranked", "horizon", "fedprox_mu", "alpha",
        "time_of_day", "use_time_encoding", "weights_path",
    }
    assert expected_keys == set(d.keys()), (
        f"Missing keys: {expected_keys - set(d.keys())}, "
        f"Extra keys: {set(d.keys()) - expected_keys}"
    )
    assert d["name"] == "fedprox_mu0.1"
    assert d["trainer_type"] == "FedRL"
    assert d["topology"] == "grid-5x5"
    assert d["n_episodes"] == 30
    assert d["fedprox_mu"] == 0.1
    assert d["alpha"] == 0.5
