"""Unit tests for BackEnd/scripts/generate_tables.py.

All tests use synthetic data — no SUMO, no trained weights, no running backend
required.  The fixture make_mock_result() produces dicts that match the
campaign results.json structure defined in 07-03-PLAN.md.

Run:
    cd BackEnd && python -m pytest tests/test_analysis.py -x -v
"""
import json
import math
import os
import sys

# Ensure BackEnd/ is on sys.path before any local imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before pyplot import

import pytest

from scripts.generate_tables import (
    generate_latex_table,
    plot_comparison_bar,
    results_to_dataframe,
    wilcoxon_compare,
    plot_convergence_curves,
    select_best_config_name,
    plot_combined_comparison,
    plot_per_topology,
    generate_ablation_table_with_pvalues,
    generate_combined_extensions_table,
)


# ---------------------------------------------------------------------------
# Test fixtures
# ---------------------------------------------------------------------------

def make_mock_result(name, trainer, topology, wait_times, travel_times, rewards):
    """Build a mock campaign result dict matching the results.json structure.

    Args:
        name: Configuration name string.
        trainer: Trainer type string (e.g. "FedRL").
        topology: Topology string (e.g. "grid-3x3").
        wait_times: List of per-run avg_waiting_time values.
        travel_times: List of per-run avg_travel_time values.
        rewards: List of per-run mean_reward values.

    Returns:
        Dict matching the campaign results.json result structure.
    """
    individual = [
        {
            "tripinfo": {"avg_waiting_time": w, "avg_travel_time": t, "throughput": 1.0, "completed_trips": 100},
            "mean_reward": r,
            "total_comm_cost": 900,
        }
        for w, t, r in zip(wait_times, travel_times, rewards)
    ]
    n = len(wait_times)
    return {
        "config": {
            "name": name,
            "trainer_type": trainer,
            "topology": topology,
            "fedprox_mu": 0.0,
            "alpha": 1.0,
            "time_of_day": False,
        },
        "training_rewards": [{"mean_reward": float(i)} for i in range(20)],
        "evaluation": {
            "config": {"trainer": trainer, "topology": topology, "n_runs": n},
            "individual_results": individual,
            "n_completed": n,
            "n_failed": 0,
            "aggregated": {
                "avg_waiting_time": {
                    "mean": sum(wait_times) / n,
                    "std": 5.0,
                    "ci_95_lower": 80.0,
                    "ci_95_upper": 90.0,
                    "min": min(wait_times),
                    "max": max(wait_times),
                },
                "avg_travel_time": {
                    "mean": sum(travel_times) / n,
                    "std": 8.0,
                    "ci_95_lower": 115.0,
                    "ci_95_upper": 130.0,
                    "min": min(travel_times),
                    "max": max(travel_times),
                },
                "mean_reward": {
                    "mean": sum(rewards) / n,
                    "std": 3.0,
                    "ci_95_lower": -50.0,
                    "ci_95_upper": -35.0,
                    "min": min(rewards),
                    "max": max(rewards),
                },
                "total_comm_cost": {
                    "mean": 900.0,
                    "std": 0.0,
                    "ci_95_lower": 900.0,
                    "ci_95_upper": 900.0,
                    "min": 900.0,
                    "max": 900.0,
                },
            },
        },
        "duration_seconds": 100.0,
    }


# Shared sample data for 10-run tests
_WAIT_A = [90, 92, 88, 91, 89, 93, 87, 90, 91, 88]
_WAIT_B = [70, 72, 68, 71, 69, 73, 67, 70, 71, 68]
_TRAVEL = [120.0] * 10
_REWARDS = [-45.0] * 10


# ---------------------------------------------------------------------------
# DataFrame tests
# ---------------------------------------------------------------------------

def test_results_to_dataframe_columns():
    """DataFrame has correct number of rows and expected columns."""
    result_a = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_b = make_mock_result("marl_baseline", "MARL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    df = results_to_dataframe([result_a, result_b])

    assert len(df) == 2, f"Expected 2 rows, got {len(df)}"

    expected_columns = [
        "name", "trainer", "topology",
        "avg_waiting_time_mean", "avg_waiting_time_std",
        "mean_reward_mean",
    ]
    for col in expected_columns:
        assert col in df.columns, f"Expected column {col!r} not in DataFrame"


def test_results_to_dataframe_skips_errors():
    """Results with evaluation=None are skipped; valid results are retained."""
    good = make_mock_result("good_config", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    bad = {
        "config": {"name": "bad_config", "trainer_type": "MARL", "topology": "grid-3x3",
                   "fedprox_mu": 0.0, "alpha": 1.0, "time_of_day": False},
        "evaluation": None,
        "duration_seconds": 0.0,
    }
    df = results_to_dataframe([good, bad])
    assert len(df) == 1, f"Expected 1 row (error result skipped), got {len(df)}"
    assert df.iloc[0]["name"] == "good_config"


# ---------------------------------------------------------------------------
# Wilcoxon tests
# ---------------------------------------------------------------------------

def test_wilcoxon_compare_significant():
    """Wilcoxon test correctly detects significant difference (p < 0.05)."""
    result_a = make_mock_result("high_wait", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_b = make_mock_result("low_wait", "MARL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)

    out = wilcoxon_compare(result_a, result_b, metric="avg_waiting_time")

    assert out["valid"] == True, f"Expected valid=True, got: {out}"
    assert out["significant"] == True, f"Expected significant=True, p={out.get('p_value')}"
    assert out["p_value"] < 0.05, f"Expected p < 0.05, got {out['p_value']}"


def test_wilcoxon_compare_insufficient_samples():
    """Wilcoxon returns valid=False with 'insufficient' reason for n < 8."""
    tiny_waits = [85.0, 90.0, 88.0]
    tiny_travels = [120.0, 125.0, 122.0]
    tiny_rewards = [-45.0, -42.0, -44.0]

    result_a = make_mock_result("tiny_a", "FedRL", "grid-3x3", tiny_waits, tiny_travels, tiny_rewards)
    result_b = make_mock_result("tiny_b", "MARL", "grid-3x3", tiny_waits, tiny_travels, tiny_rewards)

    out = wilcoxon_compare(result_a, result_b, metric="avg_waiting_time")

    assert out["valid"] == False, f"Expected valid=False for n=3, got: {out}"
    assert "insufficient" in out["reason"], f"Expected 'insufficient' in reason, got: {out['reason']}"


# ---------------------------------------------------------------------------
# LaTeX table tests
# ---------------------------------------------------------------------------

def test_generate_latex_table_has_booktabs(tmp_path):
    """LaTeX output contains required booktabs macros and pm formatting."""
    result_a = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_b = make_mock_result("marl_baseline", "MARL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    df = results_to_dataframe([result_a, result_b])

    table_path = str(tmp_path / "table.tex")
    generate_latex_table(df, "avg_waiting_time", table_path)

    assert os.path.exists(table_path), "table.tex was not created"

    content = open(table_path, encoding="utf-8").read()
    assert "\\toprule" in content, "Missing \\toprule in LaTeX output"
    assert "\\midrule" in content, "Missing \\midrule in LaTeX output"
    assert "\\bottomrule" in content, "Missing \\bottomrule in LaTeX output"
    assert "\\begin{tabular}" in content, "Missing \\begin{tabular} in LaTeX output"
    assert "\\pm" in content, "Missing \\pm (plus-minus) in LaTeX output"


# ---------------------------------------------------------------------------
# Chart tests
# ---------------------------------------------------------------------------

def test_plot_comparison_bar_creates_file(tmp_path):
    """Bar chart PNG is created and non-empty."""
    result_a = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_b = make_mock_result("marl_baseline", "MARL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    df = results_to_dataframe([result_a, result_b])

    chart_path = str(tmp_path / "chart.png")
    plot_comparison_bar(df, "avg_waiting_time", chart_path)

    assert os.path.exists(chart_path), "chart.png was not created"
    assert os.path.getsize(chart_path) > 0, "chart.png is empty"


# ---------------------------------------------------------------------------
# New analysis function tests (Task 1)
# ---------------------------------------------------------------------------

def test_plot_convergence_curves_creates_file(tmp_path):
    """Convergence curve PNG is created and non-empty from synthetic training_rewards."""
    result_a = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_b = make_mock_result("fedprox_mu0.01", "FedRL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    chart_path = str(tmp_path / "convergence.png")
    plot_convergence_curves([result_a, result_b], chart_path)
    assert os.path.exists(chart_path), "convergence.png was not created"
    assert os.path.getsize(chart_path) > 0, "convergence.png is empty"


def test_plot_convergence_curves_skips_none_rewards(tmp_path):
    """plot_convergence_curves gracefully skips results with training_rewards=None."""
    result_a = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_a["training_rewards"] = None  # No training data
    chart_path = str(tmp_path / "convergence_none.png")
    # Should not raise; file may or may not be created (0 valid curves means no file)
    plot_convergence_curves([result_a], chart_path)
    # No crash = pass (file may not exist if 0 valid curves)


def test_select_best_config_name_returns_minimum():
    """select_best_config_name returns config with lowest avg_waiting_time_mean."""
    result_a = make_mock_result("high_wait", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_b = make_mock_result("low_wait", "FedRL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    best = select_best_config_name([result_a, result_b])
    assert best == "low_wait", f"Expected 'low_wait' (lower mean), got '{best}'"


def test_select_best_config_name_empty():
    """select_best_config_name returns empty string for empty results list."""
    result = select_best_config_name([])
    assert result == "", f"Expected '' for empty input, got '{result}'"


def test_plot_combined_comparison_creates_file(tmp_path):
    """plot_combined_comparison creates non-empty PNG from multiple campaigns."""
    baseline = make_mock_result("FedRL_3x3", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    best_fedprox = make_mock_result("fedprox_mu0.01", "FedRL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    ablation_map = {"FedProx": [best_fedprox]}
    chart_path = str(tmp_path / "combined.png")
    plot_combined_comparison([baseline], ablation_map, chart_path)
    assert os.path.exists(chart_path), "combined.png was not created"
    assert os.path.getsize(chart_path) > 0, "combined.png is empty"


def test_plot_per_topology_creates_file(tmp_path):
    """plot_per_topology creates non-empty PNG filtering by topology column."""
    result_3x3 = make_mock_result("cfg_3x3", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_5x5 = make_mock_result("cfg_5x5", "FedRL", "grid-5x5", _WAIT_B, _TRAVEL, _REWARDS)
    chart_path = str(tmp_path / "per_topology.png")
    plot_per_topology([result_3x3, result_5x5], "grid-3x3", "avg_waiting_time", chart_path)
    assert os.path.exists(chart_path), "per_topology.png was not created"
    assert os.path.getsize(chart_path) > 0, "per_topology.png is empty"


def test_generate_ablation_table_with_pvalues_booktabs(tmp_path):
    """Ablation table with p-values contains booktabs markers and p= or n/a."""
    result_base = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    result_other = make_mock_result("fedprox_mu0.01", "FedRL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    table_path = str(tmp_path / "ablation_table.tex")
    generate_ablation_table_with_pvalues(
        [result_base, result_other], "fedavg_baseline", "avg_waiting_time", table_path
    )
    assert os.path.exists(table_path), "ablation_table.tex was not created"
    content = open(table_path, encoding="utf-8").read()
    assert "\\toprule" in content, "Missing \\toprule"
    assert ("p=" in content or "n/a" in content), "Missing p-value column content"


def test_generate_ablation_table_pvalue_na_when_invalid(tmp_path):
    """Ablation table shows 'n/a' when Wilcoxon is invalid (small samples)."""
    tiny_waits = [85.0, 90.0, 88.0]
    tiny_travels = [120.0, 125.0, 122.0]
    tiny_rewards = [-45.0, -42.0, -44.0]
    result_base = make_mock_result("fedavg_baseline", "FedRL", "grid-3x3", tiny_waits, tiny_travels, tiny_rewards)
    result_other = make_mock_result("fedprox_mu0.01", "FedRL", "grid-3x3", tiny_waits, tiny_travels, tiny_rewards)
    table_path = str(tmp_path / "ablation_na.tex")
    generate_ablation_table_with_pvalues(
        [result_base, result_other], "fedavg_baseline", "avg_waiting_time", table_path
    )
    content = open(table_path, encoding="utf-8").read()
    assert "n/a" in content, "Expected 'n/a' for invalid Wilcoxon, not found"


def test_generate_combined_extensions_table_bold(tmp_path):
    """Combined extensions table contains \\textbf{ for bold best value."""
    baseline = make_mock_result("FedRL_3x3", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)
    best_fedprox = make_mock_result("fedprox_mu0.01", "FedRL", "grid-3x3", _WAIT_B, _TRAVEL, _REWARDS)
    ablation_map = {"FedProx": [best_fedprox]}
    table_path = str(tmp_path / "combined_ext.tex")
    generate_combined_extensions_table([baseline], ablation_map, "avg_waiting_time", table_path)
    assert os.path.exists(table_path), "combined_ext.tex was not created"
    content = open(table_path, encoding="utf-8").read()
    assert "\\textbf{" in content, "Missing \\textbf{ for bold best value"


# ---------------------------------------------------------------------------
# generate_report tests (Task 2)
# ---------------------------------------------------------------------------

def test_generate_report_creates_md(tmp_path):
    """Report generator creates EXPERIMENT_REPORT.md with Key Findings."""
    from scripts.generate_report import generate_experiment_report

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    results = {"results": [make_mock_result("FedRL_3x3", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)]}
    (baseline_dir / "results.json").write_text(json.dumps(results))

    figures_dir = str(tmp_path / "figures")
    output = str(tmp_path / "figures" / "EXPERIMENT_REPORT.md")
    generate_experiment_report(
        campaign_dirs={"baseline": str(baseline_dir)},
        figures_dir=figures_dir,
        output_path=output,
    )
    assert os.path.exists(output), "EXPERIMENT_REPORT.md was not created"
    content = open(output).read()
    assert "Key Findings" in content, "Missing 'Key Findings' section"
    assert "SEAL Experiment Report" in content, "Missing report title"


def test_generate_report_handles_missing_ablation(tmp_path):
    """Report notes missing ablations gracefully (no crash)."""
    from scripts.generate_report import generate_experiment_report

    baseline_dir = tmp_path / "baseline"
    baseline_dir.mkdir()
    results = {"results": [make_mock_result("FedRL_3x3", "FedRL", "grid-3x3", _WAIT_A, _TRAVEL, _REWARDS)]}
    (baseline_dir / "results.json").write_text(json.dumps(results))

    figures_dir = str(tmp_path / "figures")
    output = str(tmp_path / "figures" / "EXPERIMENT_REPORT.md")
    generate_experiment_report(
        campaign_dirs={"baseline": str(baseline_dir), "fedprox": str(tmp_path / "nonexistent")},
        figures_dir=figures_dir,
        output_path=output,
    )
    content = open(output).read()
    assert ("not available" in content.lower() or "missing" in content.lower()), \
        "Expected 'not available' or 'missing' for absent ablation"
