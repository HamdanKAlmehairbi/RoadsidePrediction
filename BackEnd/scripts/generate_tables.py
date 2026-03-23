"""Statistical analysis and publishable output generation for SEAL campaign results.

Transforms raw campaign JSON results into publication-ready artifacts:
- Wilcoxon signed-rank significance tests for pairwise comparisons
- LaTeX comparison tables with booktabs formatting (mean +/- std)
- Grouped bar charts with 95% CI error bars

Usage:
    python -m scripts.generate_tables \\
        --campaign-dir BackEnd/results/campaigns/my_campaign \\
        --output-dir BackEnd/results/figures/
"""
import sys
import os

# Add BackEnd directory to sys.path so api.* imports resolve
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import argparse
import json
import logging
import math

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend — must be set before importing pyplot
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_campaign_results(campaign_dir: str) -> list:
    """Load campaign results from a directory containing results.json.

    Args:
        campaign_dir: Path to directory containing results.json.

    Returns:
        List of result dicts from the "results" key in results.json.

    Raises:
        FileNotFoundError: If results.json is not found in campaign_dir.
    """
    results_path = os.path.join(campaign_dir, "results.json")
    if not os.path.exists(results_path):
        raise FileNotFoundError(
            f"results.json not found in {campaign_dir!r}. "
            f"Expected path: {results_path}"
        )
    with open(results_path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    return data["results"]


# ---------------------------------------------------------------------------
# DataFrame builder
# ---------------------------------------------------------------------------

def results_to_dataframe(campaign_results: list) -> pd.DataFrame:
    """Flatten campaign results list into a pandas DataFrame.

    Each result becomes one row.  Results with missing or errored evaluations
    are silently skipped.

    Columns produced:
        name, trainer, topology, fedprox_mu, alpha, time_of_day,
        avg_waiting_time_mean, avg_waiting_time_std,
        avg_travel_time_mean, avg_travel_time_std,
        mean_reward_mean, mean_reward_std,
        total_comm_cost_mean, total_comm_cost_std,
        n_completed, n_failed

    Args:
        campaign_results: List of result dicts from load_campaign_results().

    Returns:
        pd.DataFrame with one row per valid result.
    """
    rows = []
    for result in campaign_results:
        evaluation = result.get("evaluation")
        if evaluation is None:
            logger.debug("Skipping result with evaluation=None: %s", result.get("config", {}).get("name"))
            continue
        if "error" in evaluation and evaluation.get("error"):
            logger.debug("Skipping errored result: %s", evaluation.get("error"))
            continue

        config = result.get("config", {})
        agg = evaluation.get("aggregated", {})

        def _mean(metric: str) -> float:
            return agg.get(metric, {}).get("mean", float("nan"))

        def _std(metric: str) -> float:
            return agg.get(metric, {}).get("std", float("nan"))

        rows.append({
            "name": config.get("name", ""),
            "trainer": config.get("trainer_type", ""),
            "topology": config.get("topology", ""),
            "fedprox_mu": config.get("fedprox_mu", 0.0),
            "alpha": config.get("alpha", 1.0),
            "time_of_day": config.get("time_of_day", False),
            "avg_waiting_time_mean": _mean("avg_waiting_time"),
            "avg_waiting_time_std": _std("avg_waiting_time"),
            "avg_travel_time_mean": _mean("avg_travel_time"),
            "avg_travel_time_std": _std("avg_travel_time"),
            "mean_reward_mean": _mean("mean_reward"),
            "mean_reward_std": _std("mean_reward"),
            "total_comm_cost_mean": _mean("total_comm_cost"),
            "total_comm_cost_std": _std("total_comm_cost"),
            "n_completed": evaluation.get("n_completed", 0),
            "n_failed": evaluation.get("n_failed", 0),
        })

    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Statistical tests
# ---------------------------------------------------------------------------

def wilcoxon_compare(
    results_a: dict,
    results_b: dict,
    metric: str = "avg_waiting_time",
) -> dict:
    """Run a Wilcoxon signed-rank test comparing two paired MC result sets.

    Extracts per-run metric values from the individual_results lists in both
    result dicts, then calls scipy.stats.wilcoxon() on the paired samples.

    For tripinfo metrics (avg_waiting_time, avg_travel_time) values are
    extracted via r["tripinfo"][metric].  For reward metrics (mean_reward)
    values are extracted directly via r[metric].

    Args:
        results_a: First result dict (from campaign results.json "results" list).
        results_b: Second result dict.
        metric: Metric name to compare.  Default "avg_waiting_time".

    Returns:
        On success:
            {"valid": True, "statistic": float, "p_value": float,
             "significant": bool, "n": int}
        On insufficient samples (n < 8):
            {"valid": False, "reason": "insufficient samples",
             "n_a": int, "n_b": int}
        On scipy error (e.g. all-zero differences):
            {"valid": False, "reason": str(exc)}
    """
    _TRIPINFO_METRICS = {"avg_waiting_time", "avg_travel_time", "throughput"}

    def _extract_values(result: dict) -> list:
        individual = result.get("evaluation", {}).get("individual_results", [])
        if metric in _TRIPINFO_METRICS:
            return [r["tripinfo"][metric] for r in individual if "tripinfo" in r]
        else:
            return [r[metric] for r in individual if metric in r]

    eval_a = results_a.get("evaluation", {})
    eval_b = results_b.get("evaluation", {})
    n_completed_a = eval_a.get("n_completed", 0) if eval_a else 0
    n_completed_b = eval_b.get("n_completed", 0) if eval_b else 0

    if n_completed_a < 8 or n_completed_b < 8:
        values_a = _extract_values(results_a)
        values_b = _extract_values(results_b)
        return {
            "valid": False,
            "reason": "insufficient samples",
            "n_a": len(values_a),
            "n_b": len(values_b),
        }

    values_a = _extract_values(results_a)
    values_b = _extract_values(results_b)

    try:
        stat, p = stats.wilcoxon(values_a, values_b)
        return {
            "valid": True,
            "statistic": float(stat),
            "p_value": float(p),
            "significant": p < 0.05,
            "n": min(len(values_a), len(values_b)),
        }
    except ValueError as exc:
        return {"valid": False, "reason": str(exc)}


# ---------------------------------------------------------------------------
# LaTeX table generation
# ---------------------------------------------------------------------------

def generate_latex_table(
    df: pd.DataFrame,
    metric: str,
    output_path: str,
) -> None:
    """Write a booktabs LaTeX table comparing metric values across conditions.

    Rows represent configuration names; columns represent trainer types.
    Cell values are formatted as "{mean:.1f} $\\pm$ {std:.1f}".

    Uses hand-crafted LaTeX with \\toprule, \\midrule, \\bottomrule — NOT
    pandas .to_latex() — for full booktabs compliance.

    Args:
        df: DataFrame from results_to_dataframe().
        metric: Base metric name (e.g. "avg_waiting_time").
        output_path: File path for the output .tex file.
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in df.columns or std_col not in df.columns:
        raise ValueError(f"Metric columns {mean_col!r} / {std_col!r} not found in DataFrame")

    # Pivot: rows = name, columns = trainer
    trainers = df["trainer"].unique().tolist()
    names = df["name"].unique().tolist()

    # Build a lookup dict: (name, trainer) -> formatted string
    lookup: dict = {}
    for _, row in df.iterrows():
        key = (row["name"], row["trainer"])
        mean_val = row[mean_col]
        std_val = row[std_col]
        lookup[key] = f"{mean_val:.1f} $\\pm$ {std_val:.1f}"

    # Build LaTeX column spec: l + c per trainer
    col_spec = "l" + "c" * len(trainers)

    lines = [
        "\\begin{tabular}{" + col_spec + "}",
        "\\toprule",
        "Configuration & " + " & ".join(trainers) + " \\\\",
        "\\midrule",
    ]

    for name in names:
        cells = [name]
        for trainer in trainers:
            cells.append(lookup.get((name, trainer), "---"))
        lines.append(" & ".join(cells) + " \\\\")

    lines += [
        "\\bottomrule",
        "\\end{tabular}",
    ]

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines) + "\n")

    logger.info("LaTeX table written to %s", output_path)


# ---------------------------------------------------------------------------
# Bar chart generation
# ---------------------------------------------------------------------------

def plot_comparison_bar(
    df: pd.DataFrame,
    metric: str,
    output_path: str,
    group_by: str = "name",
) -> None:
    """Generate a grouped bar chart with 95% CI error bars.

    Groups are defined by the group_by column (default "name"); bars within
    each group are coloured by trainer type.  Error bars use:
        CI = 1.96 * std / sqrt(n)
    where n is taken from the "n_completed" column.

    Args:
        df: DataFrame from results_to_dataframe().
        metric: Base metric name (e.g. "avg_waiting_time").
        output_path: File path for the output PNG.
        group_by: Column to use for x-axis groups.  Default "name".
    """
    mean_col = f"{metric}_mean"
    std_col = f"{metric}_std"

    if mean_col not in df.columns or std_col not in df.columns:
        raise ValueError(f"Metric columns {mean_col!r} / {std_col!r} not found in DataFrame")

    groups = df[group_by].unique().tolist()
    trainers = df["trainer"].unique().tolist()
    n_groups = len(groups)
    n_trainers = len(trainers)

    fig, ax = plt.subplots(figsize=(10, 5))

    bar_width = 0.8 / max(n_trainers, 1)
    x = np.arange(n_groups)

    for i, trainer in enumerate(trainers):
        subset = df[df["trainer"] == trainer].set_index(group_by)
        means = []
        cis = []
        for group in groups:
            if group in subset.index:
                row = subset.loc[group]
                mean_val = row[mean_col]
                std_val = row[std_col]
                n = row.get("n_completed", 10) if isinstance(row, pd.Series) else 10
                ci = 1.96 * std_val / math.sqrt(max(n, 1))
            else:
                mean_val = 0.0
                ci = 0.0
            means.append(mean_val)
            cis.append(ci)

        offsets = x + i * bar_width - (n_trainers - 1) * bar_width / 2
        ax.bar(
            offsets,
            means,
            width=bar_width,
            label=trainer,
            yerr=cis,
            capsize=4,
            error_kw={"elinewidth": 1.5},
        )

    # Axis labels — add units for time metrics
    _TIME_METRICS = {"avg_waiting_time", "avg_travel_time"}
    ylabel = metric.replace("_", " ").title()
    if metric in _TIME_METRICS:
        ylabel += " (s)"
    ax.set_ylabel(ylabel)
    ax.set_xlabel(group_by.replace("_", " ").title())
    ax.set_xticks(x)
    ax.set_xticklabels(groups, rotation=15, ha="right")
    ax.legend()
    ax.set_title(f"{ylabel} by {group_by.replace('_', ' ').title()}")

    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()

    logger.info("Bar chart written to %s", output_path)


# ---------------------------------------------------------------------------
# All-outputs generator
# ---------------------------------------------------------------------------

def generate_all_outputs(
    campaign_dir: str,
    output_dir: str = None,
) -> None:
    """Generate all publishable outputs from a campaign results directory.

    Produces:
        - LaTeX tables for avg_waiting_time and avg_travel_time
        - Bar charts for avg_waiting_time, avg_travel_time, mean_reward

    Args:
        campaign_dir: Path to campaign directory containing results.json.
        output_dir: Output directory for figures and tables.  Defaults to
            BackEnd/results/figures/ (relative to this file's location).
    """
    if output_dir is None:
        output_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "results", "figures")
        )

    os.makedirs(output_dir, exist_ok=True)

    logger.info("Loading campaign results from %s", campaign_dir)
    campaign_results = load_campaign_results(campaign_dir)
    df = results_to_dataframe(campaign_results)

    if df.empty:
        logger.warning("No valid results found — no outputs generated")
        return

    logger.info("Building outputs for %d configurations in %s", len(df), output_dir)

    # LaTeX tables
    for metric in ("avg_waiting_time", "avg_travel_time"):
        table_path = os.path.join(output_dir, f"table_{metric}.tex")
        generate_latex_table(df, metric, table_path)
        logger.info("Generated: %s", table_path)

    # Bar charts
    for metric in ("avg_waiting_time", "avg_travel_time", "mean_reward"):
        chart_path = os.path.join(output_dir, f"chart_{metric}.png")
        plot_comparison_bar(df, metric, chart_path)
        logger.info("Generated: %s", chart_path)


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """CLI entry point for generate_tables.py."""
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    parser = argparse.ArgumentParser(
        description="Generate publication-ready tables and charts from SEAL campaign results."
    )
    parser.add_argument(
        "--campaign-dir",
        required=True,
        help="Path to campaign results directory (must contain results.json).",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="Output directory for figures and tables (default: BackEnd/results/figures/).",
    )
    parser.add_argument(
        "--metric",
        default="avg_waiting_time",
        help="Metric for single chart/table mode (default: avg_waiting_time).",
    )
    parser.add_argument(
        "--format",
        choices=["all", "table", "chart"],
        default="all",
        help="Output format: all, table only, or chart only (default: all).",
    )
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.normpath(
            os.path.join(os.path.dirname(__file__), "..", "results", "figures")
        )

    campaign_results = load_campaign_results(args.campaign_dir)
    df = results_to_dataframe(campaign_results)

    if df.empty:
        logger.warning("No valid results to process")
        return

    if args.format == "all":
        generate_all_outputs(args.campaign_dir, args.output_dir)
    elif args.format == "table":
        table_path = os.path.join(args.output_dir, f"table_{args.metric}.tex")
        generate_latex_table(df, args.metric, table_path)
    elif args.format == "chart":
        chart_path = os.path.join(args.output_dir, f"chart_{args.metric}.png")
        plot_comparison_bar(df, args.metric, chart_path)


if __name__ == "__main__":
    main()
