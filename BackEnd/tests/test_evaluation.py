"""Integration tests for the SEAL evaluation framework.

Tests cover:
- Package-level imports (smoke test)
- TRAINERS / TOPOLOGIES constants
- compute_tripinfo_metrics() from synthetic XML
- _aggregate_metrics() mean / std / CI computation
- compute_transfer_gap() gap = home - transfer
- Store save/load round-trip
- POST /api/evaluate returns 202
"""
import sys
import os

# Ensure BackEnd root is on the path so `api` is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest

# ---------------------------------------------------------------------------
# Optional SUMO availability guard (used for tests that require SUMO)
# ---------------------------------------------------------------------------

def _sumo_available() -> bool:
    """Return True if SUMO / TraCI can be imported."""
    try:
        import traci  # noqa: F401
        return True
    except ImportError:
        pass
    # Also check SUMO_HOME env var as a secondary signal
    return bool(os.environ.get("SUMO_HOME"))


SUMO_AVAILABLE = _sumo_available()


# ---------------------------------------------------------------------------
# Test 1 — Smoke test: all public API names importable
# ---------------------------------------------------------------------------

def test_imports():
    """Verify the evaluation package exports all expected public names."""
    from api.evaluation import (  # noqa: F401
        run_trial, TrialResult, resolve_weights_path, TRAINERS, TOPOLOGIES,
        run_fixed_time_trial, run_max_pressure_trial, run_rl_trial,
        compute_tripinfo_metrics, compute_trial_metrics, metrics_to_dict,
        TripinfoMetrics, TrialMetrics,
        build_transfer_matrix, compute_transfer_gap, transfer_matrix_to_dict,
        TransferResult,
        run_monte_carlo, run_full_campaign, campaign_to_dict,
        MCConfig, MCAggregatedResult,
        save_evaluation, load_evaluation, list_evaluations,
    )


# ---------------------------------------------------------------------------
# Test 2 — TRAINERS constant contains all 5 trainer types
# ---------------------------------------------------------------------------

def test_trainers_constant():
    """All 5 trainer types must be present in TRAINERS."""
    from api.evaluation import TRAINERS

    expected = {"FedRL", "MARL", "SARL", "fixed-time", "max-pressure"}
    assert set(TRAINERS) == expected, f"TRAINERS mismatch: {TRAINERS}"
    assert len(TRAINERS) == 5, f"Expected 5 trainers, got {len(TRAINERS)}"


# ---------------------------------------------------------------------------
# Test 3 — TOPOLOGIES constant contains all 3 grid topologies
# ---------------------------------------------------------------------------

def test_topologies_constant():
    """All 3 topologies must be present in TOPOLOGIES."""
    from api.evaluation import TOPOLOGIES

    assert "grid-3x3" in TOPOLOGIES, "'grid-3x3' missing from TOPOLOGIES"
    assert "grid-5x5" in TOPOLOGIES, "'grid-5x5' missing from TOPOLOGIES"
    assert "grid-7x7" in TOPOLOGIES, "'grid-7x7' missing from TOPOLOGIES"
    assert len(TOPOLOGIES) >= 3, f"Expected at least 3 topologies, got {len(TOPOLOGIES)}"


# ---------------------------------------------------------------------------
# Test 4 — compute_tripinfo_metrics() parses synthetic XML correctly
# ---------------------------------------------------------------------------

def test_compute_tripinfo_metrics(tmp_path):
    """Parse a synthetic tripinfo XML and verify averaging."""
    from api.evaluation import compute_tripinfo_metrics

    tripinfo_xml = tmp_path / "tripinfo.xml"
    tripinfo_xml.write_text(
        '<?xml version="1.0"?>\n'
        "<tripinfos>\n"
        '  <tripinfo id="veh_0" depart="0.0" departDelay="0.0" arrival="100.0"'
        ' duration="100.0" routeLength="500.0" waitingTime="10.0"'
        ' waitingCount="2" rerouteNo="0"/>\n'
        '  <tripinfo id="veh_1" depart="5.0" departDelay="2.0" arrival="120.0"'
        ' duration="115.0" routeLength="600.0" waitingTime="20.0"'
        ' waitingCount="3" rerouteNo="0"/>\n'
        '  <tripinfo id="veh_2" depart="10.0" departDelay="1.0" arrival="80.0"'
        ' duration="70.0" routeLength="400.0" waitingTime="5.0"'
        ' waitingCount="1" rerouteNo="0"/>\n'
        "</tripinfos>\n",
        encoding="utf-8",
    )

    result = compute_tripinfo_metrics(str(tripinfo_xml))

    # avg_waiting_time = (10 + 20 + 5) / 3 = 11.666...
    assert abs(result.avg_waiting_time - 11.6667) < 0.01, (
        f"avg_waiting_time expected ~11.667, got {result.avg_waiting_time}"
    )

    # avg_travel_time = (100 + 115 + 70) / 3 = 95.0
    assert abs(result.avg_travel_time - 95.0) < 0.01, (
        f"avg_travel_time expected ~95.0, got {result.avg_travel_time}"
    )

    # completed_trips = 3
    assert result.completed_trips == 3, (
        f"completed_trips expected 3, got {result.completed_trips}"
    )


def test_compute_tripinfo_metrics_missing_file():
    """Missing file returns zero-valued TripinfoMetrics without raising."""
    from api.evaluation import compute_tripinfo_metrics

    result = compute_tripinfo_metrics("/nonexistent/path/tripinfo.xml")
    assert result.completed_trips == 0
    assert result.avg_waiting_time == 0.0


# ---------------------------------------------------------------------------
# Test 5 — _aggregate_metrics() computes mean / std / CI correctly
# ---------------------------------------------------------------------------

def test_aggregate_metrics():
    """_aggregate_metrics computes mean -1.5, std 0.5, and valid 95% CI."""
    from api.evaluation.monte_carlo import _aggregate_metrics
    from api.evaluation import TrialMetrics, TripinfoMetrics

    # Build three TrialMetrics with known mean_reward values
    def _make_metrics(mean_reward: float) -> TrialMetrics:
        return TrialMetrics(
            tripinfo=TripinfoMetrics(
                avg_waiting_time=0.0,
                avg_travel_time=0.0,
                throughput=1.0,
                completed_trips=0,
            ),
            mean_reward=mean_reward,
            total_reward=mean_reward,
            comm_costs={},
            total_comm_cost=0,
            trainer="fixed-time",
            topology="grid-3x3",
            seed=42,
        )

    metrics_list = [_make_metrics(-2.0), _make_metrics(-1.5), _make_metrics(-1.0)]
    result = _aggregate_metrics(metrics_list)

    assert "mean_reward" in result, "mean_reward key missing from aggregated result"

    mr = result["mean_reward"]

    # mean should be -1.5
    assert abs(mr["mean"] - (-1.5)) < 1e-4, f"mean expected -1.5, got {mr['mean']}"

    # std of [-2.0, -1.5, -1.0] = 0.5
    assert abs(mr["std"] - 0.5) < 1e-4, f"std expected 0.5, got {mr['std']}"

    # CI bounds should straddle the mean
    assert mr["ci_95_lower"] < mr["mean"], (
        f"ci_95_lower ({mr['ci_95_lower']}) should be < mean ({mr['mean']})"
    )
    assert mr["ci_95_upper"] > mr["mean"], (
        f"ci_95_upper ({mr['ci_95_upper']}) should be > mean ({mr['mean']})"
    )


# ---------------------------------------------------------------------------
# Test 6 — compute_transfer_gap() gap = home_reward - transfer_reward
# ---------------------------------------------------------------------------

def test_transfer_gap_computation():
    """Transfer gap should equal home_reward - transfer_reward = 1.0."""
    from api.evaluation import compute_transfer_gap, TransferResult, TrialMetrics, TripinfoMetrics

    def _make_transfer_result(is_transfer: bool, mean_reward: float) -> TransferResult:
        metrics = TrialMetrics(
            tripinfo=TripinfoMetrics(),
            mean_reward=mean_reward,
            total_reward=mean_reward,
            comm_costs={},
            total_comm_cost=0,
            trainer="FedRL",
            topology="grid-3x3",
            seed=42,
        )
        return TransferResult(
            train_topology="grid-3x3",
            test_topology="grid-5x5" if is_transfer else "grid-3x3",
            trainer="FedRL",
            metrics=metrics,
            is_transfer=is_transfer,
        )

    # 3 home results (is_transfer=False) with mean_reward = -1.0
    home_results = [_make_transfer_result(False, -1.0) for _ in range(3)]
    # 6 transfer results (is_transfer=True) with mean_reward = -2.0
    transfer_results = [_make_transfer_result(True, -2.0) for _ in range(6)]

    all_results = home_results + transfer_results
    gap_list = compute_transfer_gap(all_results)

    assert len(gap_list) == 1, f"Expected 1 gap entry (FedRL), got {len(gap_list)}"
    gap_entry = gap_list[0]

    # gap = home_perf - transfer_perf = -1.0 - (-2.0) = 1.0
    assert abs(gap_entry["gap"] - 1.0) < 1e-4, (
        f"gap expected 1.0, got {gap_entry['gap']}"
    )
    assert gap_entry["trainer"] == "FedRL"
    assert abs(gap_entry["home_reward"] - (-1.0)) < 1e-4
    assert abs(gap_entry["transfer_reward"] - (-2.0)) < 1e-4


# ---------------------------------------------------------------------------
# Test 7 — Store save/load round-trip
# ---------------------------------------------------------------------------

def test_store_roundtrip(tmp_path, monkeypatch):
    """save_evaluation then load_evaluation should return identical data."""
    import api.evaluation.store as store_module

    # Redirect RESULTS_DIR to tmp_path
    monkeypatch.setattr(store_module, "RESULTS_DIR", str(tmp_path))

    result_data = {"status": "complete", "mean_reward": -1.5, "trainer": "FedRL"}
    save_path = store_module.save_evaluation("test_job_1", result_data)

    # File should exist
    assert os.path.isfile(save_path), f"Saved file not found at {save_path}"

    # Round-trip load
    loaded = store_module.load_evaluation("test_job_1")
    assert loaded is not None, "load_evaluation returned None"
    assert loaded["status"] == "complete"
    assert loaded["mean_reward"] == -1.5

    # list_evaluations should include our job
    listing = store_module.list_evaluations()
    job_ids = [entry.get("job_id") for entry in listing]
    assert "test_job_1" in job_ids, f"test_job_1 not in listing: {job_ids}"


def test_load_evaluation_missing_returns_none(tmp_path, monkeypatch):
    """load_evaluation of a nonexistent job returns None."""
    import api.evaluation.store as store_module

    monkeypatch.setattr(store_module, "RESULTS_DIR", str(tmp_path))

    result = store_module.load_evaluation("nonexistent_job_xyz")
    assert result is None


# ---------------------------------------------------------------------------
# Test 8 — POST /api/evaluate returns 202
# ---------------------------------------------------------------------------

@pytest.mark.skipif(not SUMO_AVAILABLE, reason="SUMO/TraCI not available in this environment")
def test_api_evaluate_endpoint_sumo():
    """POST /api/evaluate with SUMO available returns 202 and job_id."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app)
    response = client.post(
        "/api/evaluate",
        json={"trainers": ["fixed-time"], "topologies": ["grid-3x3"], "n_runs": 1},
    )
    assert response.status_code == 202, (
        f"Expected 202, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert "job_id" in data, f"job_id missing from response: {data}"
    assert data["status"] == "running", f"status expected 'running', got {data['status']}"


def test_api_evaluate_endpoint():
    """POST /api/evaluate returns 202 without SUMO (job starts async)."""
    from fastapi.testclient import TestClient
    from api.main import app

    client = TestClient(app, raise_server_exceptions=False)
    response = client.post(
        "/api/evaluate",
        json={"trainers": ["fixed-time"], "topologies": ["grid-3x3"], "n_runs": 1},
    )
    # The endpoint always returns 202 immediately — actual work happens in background
    assert response.status_code == 202, (
        f"Expected 202, got {response.status_code}: {response.text}"
    )
    data = response.json()
    assert "job_id" in data, f"job_id missing from response: {data}"
    assert data["status"] == "running", f"status expected 'running', got {data['status']}"
