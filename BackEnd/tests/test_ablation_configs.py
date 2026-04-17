"""Unit tests for extension ablation config builders.

Validates that each ablation study's config builder returns the correct
parameter values, respects overrides (topology, n_episodes), and enforces
important constraints (e.g. alpha=0.0 exclusion per research pitfall 6).

Usage:
    cd BackEnd && python -m pytest tests/test_ablation_configs.py -x -v
"""
import os
import sys

# ---------------------------------------------------------------------------
# Ensure BackEnd/ is on sys.path so `scripts.*` imports work
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.run_extension_ablation import (
    build_cooperative_configs,
    build_fedprox_configs,
    build_time_of_day_configs,
)


# ---------------------------------------------------------------------------
# FedProx ablation tests
# ---------------------------------------------------------------------------


def test_fedprox_configs_count_and_mu_values():
    """FedProx ablation returns 3 configs with mu in {0.0, 0.01, 0.1}."""
    configs = build_fedprox_configs()

    assert len(configs) == 3, f"Expected 3 configs, got {len(configs)}"

    mu_values = [c.fedprox_mu for c in configs]
    assert mu_values == [0.0, 0.01, 0.1], (
        f"Expected mu values [0.0, 0.01, 0.1], got {mu_values}"
    )

    trainer_types = [c.trainer_type for c in configs]
    assert all(t == "FedRL" for t in trainer_types), (
        f"All FedProx configs must use FedRL trainer, got {trainer_types}"
    )


# ---------------------------------------------------------------------------
# Cooperative reward ablation tests
# ---------------------------------------------------------------------------


def test_cooperative_configs_no_zero_alpha():
    """Cooperative ablation returns 3 configs with alpha in {1.0, 0.5, 0.1}.

    Specifically verifies that alpha=0.0 is NOT present — it creates a
    degenerate zero reward signal (research pitfall 6).
    """
    configs = build_cooperative_configs()

    assert len(configs) == 3, f"Expected 3 configs, got {len(configs)}"

    alpha_values = [c.alpha for c in configs]
    assert alpha_values == [1.0, 0.5, 0.1], (
        f"Expected alpha values [1.0, 0.5, 0.1], got {alpha_values}"
    )

    # Safety check: alpha=0.0 must not appear (pitfall 6 — degenerate reward)
    assert 0.0 not in alpha_values, (
        "alpha=0.0 must NOT be in cooperative configs — degenerate zero reward signal"
    )


# ---------------------------------------------------------------------------
# Time-of-day ablation tests
# ---------------------------------------------------------------------------


def test_time_of_day_configs_encoding_pairing():
    """ToD ablation returns 2 configs with correct time_of_day/use_time_encoding pairing."""
    configs = build_time_of_day_configs()

    assert len(configs) == 2, f"Expected 2 configs, got {len(configs)}"

    # Config 0: fixed demand baseline — no ToD, no encoding
    assert configs[0].time_of_day is False, (
        f"Config 0 (fixed_demand) must have time_of_day=False, "
        f"got {configs[0].time_of_day}"
    )
    assert configs[0].use_time_encoding is False, (
        f"Config 0 (fixed_demand) must have use_time_encoding=False, "
        f"got {configs[0].use_time_encoding}"
    )

    # Config 1: full ToD treatment — demand varies AND sin/cos time features in obs
    assert configs[1].time_of_day is True, (
        f"Config 1 (tod_with_encoding) must have time_of_day=True, "
        f"got {configs[1].time_of_day}"
    )
    assert configs[1].use_time_encoding is True, (
        f"Config 1 (tod_with_encoding) must have use_time_encoding=True, "
        f"got {configs[1].use_time_encoding}"
    )


# ---------------------------------------------------------------------------
# Override tests
# ---------------------------------------------------------------------------


def test_configs_respect_topology_override():
    """Config builders propagate topology override to all configs."""
    configs = build_fedprox_configs(topology="grid-5x5")

    assert all(c.topology == "grid-5x5" for c in configs), (
        f"All configs should have topology='grid-5x5', "
        f"got {[c.topology for c in configs]}"
    )


def test_configs_respect_episodes_override():
    """Config builders propagate n_episodes override to all configs."""
    configs = build_fedprox_configs(n_episodes=30)

    assert all(c.n_episodes == 30 for c in configs), (
        f"All configs should have n_episodes=30, "
        f"got {[c.n_episodes for c in configs]}"
    )
