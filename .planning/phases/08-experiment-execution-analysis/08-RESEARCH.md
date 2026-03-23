# Phase 8: Experiment Execution & Analysis - Research

**Researched:** 2026-03-23
**Domain:** SEAL experiment execution, matplotlib/scipy analysis, publication artifact generation
**Confidence:** HIGH

## Summary

Phase 8 is primarily an execution phase — all campaign infrastructure exists from Phase 7. The three CLI scripts (`run_campaign.py`, `run_extension_ablation.py`, `generate_tables.py`) are fully implemented, tested, and verified. No new infrastructure is required to run baseline, FedProx, cooperative, or time-of-day campaigns.

The new code in Phase 8 is a thin layer of analysis tools: a training convergence curve plotter, a cross-ablation combined comparison chart, per-topology breakdown charts, a combined extensions LaTeX table, and a summary report generator. All of these consume existing `results/campaigns/*/results.json` files. The primary execution risk is wall-clock time: 4 campaigns x 2 topologies x training (50 episodes each) + 10 MC seeds = significant SUMO compute. The dry-run gate (D-05) and baseline checkpoint (D-06) directly address this risk.

A key structural finding: `run_extension_ablation.py` defaults to `--topology grid-3x3` only. The CONTEXT decisions (D-01) specify both grid-3x3 AND grid-5x5 for the full scope, which means ablation runs must be invoked with `--topology grid-3x3` then `--topology grid-5x5` separately, or the script must be updated to accept `--topologies` (plural) to match the baseline runner.

**Primary recommendation:** Plan three waves: (1) dry-run smoke test, (2) baseline execution + user checkpoint, (3) ablations + full analysis generation. New analysis code (convergence curves, combined table, report) is pure Python — unit-testable with synthetic JSON without requiring SUMO.

---

<user_constraints>
## User Constraints (from CONTEXT.md)

### Locked Decisions

**Execution Order & Scope**
- D-01: Topologies: grid-3x3 AND grid-5x5 (matches full SEAL paper scope)
- D-02: Training episodes: 50 per config (Phase 7 default)
- D-03: MC seeds: 10 per config (already configured in scripts)
- D-04: Execution order: baseline first, then all 3 ablations sequentially
- D-05: Dry-run (1 seed) baseline only before committing to full runs
- D-06: After baseline completes, checkpoint for user review before proceeding to ablations

**Failure Handling**
- D-07: On failure: log error in CampaignResult, skip failed config, continue remaining configs
- D-08: No automatic retries — failed configs are logged for manual investigation

**Tables (LaTeX booktabs, mean +/- std)**
- D-09: Baseline comparison table: FedRL vs MARL vs SARL vs fixed-time vs max-pressure (avg wait time, travel time)
- D-10: FedProx ablation table: mu=0.0 vs 0.01 vs 0.1 with Wilcoxon p-values
- D-11: Cooperative ablation table: alpha=1.0 vs 0.5 vs 0.1 with Wilcoxon p-values
- D-12: Combined extensions table: baseline vs best-FedProx vs best-cooperative vs ToD — single summary comparing all extensions

**Charts**
- D-13: Grouped bar charts per ablation (already in generate_tables.py) — avg wait time with 95% CI error bars
- D-14: Training convergence curves: episode reward over time for each config (NEW — not in generate_tables.py)
- D-15: Combined comparison chart: baseline + best config from each ablation side-by-side
- D-16: Per-topology breakdown: separate charts for grid-3x3 and grid-5x5 showing extension scaling

**Summary Report**
- D-17: Auto-generated markdown report pulling together all tables, charts, significance results, and key takeaways — ready to paste into paper draft
- D-18: Report saved to BackEnd/results/figures/EXPERIMENT_REPORT.md

### Claude's Discretion
- Script orchestration approach (wrapper script vs plan tasks calling existing CLIs)
- Training convergence curve implementation details (matplotlib style, smoothing)
- Report markdown formatting and narrative structure
- "Best config" selection logic for combined table/chart (lowest avg wait time)

### Deferred Ideas (OUT OF SCOPE)
- grid-7x7 experiments — no example weights, very slow, defer to v2
- Transfer matrix analysis (train on grid-3x3, test on grid-5x5) — separate analysis phase
- Interactive dashboard for exploring results — frontend phase
</user_constraints>

---

<phase_requirements>
## Phase Requirements

| ID | Description | Research Support |
|----|-------------|------------------|
| EXP-01 | Baseline reproduction run (FedRL/MARL/SARL/fixed-time/max-pressure on grid-3x3 + grid-5x5, 10 MC seeds) | `run_campaign.py --campaign baseline --topologies grid-3x3 grid-5x5` ready; EXAMPLE_WEIGHTS_MAP confirmed for all 6 trainer/topology pairs |
| EXP-02 | FedProx ablation run (mu in {0.0, 0.01, 0.1}, both topologies, 10 MC seeds, fresh training) | `run_extension_ablation.py --ablation fedprox` ready; gap: single `--topology` flag — needs two invocations or script update for both topologies |
| EXP-03 | Cooperative ablation run (alpha in {1.0, 0.5, 0.1}, both topologies, 10 MC seeds, fresh training) | `run_extension_ablation.py --ablation cooperative` ready; same topology gap as EXP-02 |
| EXP-04 | Time-of-day ablation run (fixed vs ToD+encoding, both topologies, 10 MC seeds, fresh training) | `run_extension_ablation.py --ablation time-of-day` ready; same topology gap as EXP-02 |
| EXP-05 | Statistical analysis & figures (LaTeX tables, bar charts, convergence curves, Wilcoxon p-values, summary report) | `generate_tables.py` provides tables + bar charts; training convergence curve, combined table, per-topology charts, and summary report are new code needed |
</phase_requirements>

---

## Standard Stack

### Core (all already installed)
| Library | Version | Purpose | Why Standard |
|---------|---------|---------|--------------|
| matplotlib | 3.x (Agg backend) | All chart generation | Already used in generate_tables.py; Agg avoids display dependency |
| scipy.stats | 1.x | wilcoxon_compare() | Already imported in generate_tables.py; authoritative Wilcoxon implementation |
| pandas | 2.x | DataFrame operations | Already in results_to_dataframe(); standard for tabular data |
| numpy | 1.x / 2.x | Array operations for smoothing | Used in plot_comparison_bar(); needed for moving average in convergence curves |

### New Analysis Dependencies (no new installs — all in existing requirements)
| Library | Purpose |
|---------|---------|
| json (stdlib) | Loading results.json files across campaigns |
| os / pathlib (stdlib) | Output directory management |
| statistics (stdlib) | Cross-ablation min selection for "best config" logic |

**Installation:** No new packages required. All dependencies were added in Phase 7.

---

## Architecture Patterns

### Results Directory Layout (current)
```
BackEnd/results/
├── campaigns/              # Created by run_campaign.py / run_extension_ablation.py
│   ├── baseline/
│   │   ├── results.json    # [{config, training_rewards, evaluation, ...}, ...]
│   │   └── config.json
│   ├── fedprox-ablation/
│   │   ├── results.json
│   │   └── config.json
│   ├── cooperative-ablation/
│   │   ├── results.json
│   │   └── config.json
│   └── tod-ablation/
│       ├── results.json
│       └── config.json
├── figures/                # Output of generate_tables.py + new analysis scripts
│   ├── table_avg_waiting_time.tex
│   ├── table_avg_travel_time.tex
│   ├── chart_avg_waiting_time.png
│   ├── chart_convergence_fedprox.png    # NEW (D-14)
│   ├── chart_combined_comparison.png   # NEW (D-15)
│   ├── chart_topology_3x3.png          # NEW (D-16)
│   ├── chart_topology_5x5.png          # NEW (D-16)
│   └── EXPERIMENT_REPORT.md            # NEW (D-17/D-18)
├── evaluations/            # Phase 3 REST API eval store (pre-existing)
└── tripinfo/               # SUMO trip XML files (pre-existing)
```

### Pattern 1: Execution-then-Analysis
**What:** Run campaign scripts to populate `results/campaigns/`, then run analysis scripts that consume those files. Two distinct stages with a checkpoint between them.
**When to use:** Whenever SUMO runs are expensive — separates the long-running compute from the analysis.
**Example:**
```bash
# Stage 1: Run (produces results.json)
cd BackEnd && python scripts/run_campaign.py --campaign baseline --dry-run 1

# Stage 2: Analyse (consumes results.json, produces .tex and .png)
cd BackEnd && python scripts/generate_tables.py \
    --campaign-dir results/campaigns/baseline \
    --output-dir results/figures/
```

### Pattern 2: Training Rewards from CampaignResult
**What:** `CampaignResult.training_rewards` is a list of dicts stored in `results.json`. For ablation runs (where weights are trained fresh), this list contains per-episode reward data.
**When to use:** Building convergence curves (D-14). Baseline runs with pre-loaded weights have `training_rewards=None` — must be guarded.
**Example:**
```python
# Source: BackEnd/api/evaluation/campaign_config.py, CampaignResult
result = results[0]
rewards = result.get("training_rewards")  # None if weights_path was pre-set
if rewards:
    episode_nums = list(range(len(rewards)))
    reward_values = [r.get("mean_reward", 0) for r in rewards]
```

### Pattern 3: Cross-Ablation Result Aggregation
**What:** "Best config" for combined table (D-12/D-15) is selected by lowest `avg_waiting_time_mean` across all configs in an ablation.
**When to use:** Generating the combined extensions summary table and combined comparison chart.
**Example:**
```python
# Load all ablation results, find best config by metric
def select_best(results_list, metric="avg_waiting_time_mean"):
    df = results_to_dataframe(results_list)
    return df.loc[df[metric].idxmin()]
```

### Pattern 4: Per-Topology Filtering
**What:** Filter the unified DataFrame by the `topology` column to produce per-topology charts.
**When to use:** D-16 per-topology breakdown charts. Ablation scripts save both topologies into the same campaign directory.
**Example:**
```python
df_3x3 = df[df["topology"] == "grid-3x3"]
df_5x5 = df[df["topology"] == "grid-5x5"]
plot_comparison_bar(df_3x3, "avg_waiting_time", "chart_topology_3x3.png")
```

### Pattern 5: Ablation Tables with Wilcoxon p-values (D-10, D-11)
**What:** Standard `generate_latex_table()` shows mean±std but not p-values. Ablation tables (D-10, D-11) need an extra row or column with `p=X.XXX` from `wilcoxon_compare()`.
**When to use:** Whenever comparing ablation conditions. The Wilcoxon test needs `n_completed >= 8` per the existing guard.
**How:** Extend LaTeX table generation with an optional p-value annotation row, or embed p-values in cell footnotes.

### Anti-Patterns to Avoid
- **Running both topologies in the same `run_extension_ablation.py` call:** The script only accepts `--topology` (singular) — not `--topologies` (plural) like `run_campaign.py`. Either call it twice, or add a `--topologies` flag.
- **Loading training_rewards for baseline configs:** These have `training_rewards=None` because baseline uses pre-loaded weights. Attempting to plot convergence curves for them will fail unless guarded.
- **Cross-contaminating ablation results.json:** Each ablation saves to its own named campaign directory. Loading the wrong directory into `generate_tables.py` produces silently wrong tables.
- **Calling `generate_all_outputs()` with mixed-campaign data:** The existing function expects all rows to be from the same campaign type. For the combined table (D-12), write a dedicated function that loads from multiple directories.

---

## Don't Hand-Roll

| Problem | Don't Build | Use Instead | Why |
|---------|-------------|-------------|-----|
| Wilcoxon significance testing | Custom permutation test | `scipy.stats.wilcoxon()` in `wilcoxon_compare()` | Already implemented, tested, handles edge cases |
| LaTeX table formatting | String template with manual escaping | `generate_latex_table()` in generate_tables.py | Already produces booktabs-compliant output |
| Bar chart with CI error bars | Raw matplotlib bar loop | `plot_comparison_bar()` in generate_tables.py | Already handles 95% CI, grouped bars, Agg backend |
| Campaign JSON loading | Custom JSON parser | `load_campaign_results()` + `results_to_dataframe()` | Already flattens nested aggregated/individual structure |
| Moving average smoothing | Custom rolling implementation | `np.convolve(rewards, np.ones(k)/k, mode='valid')` | One-liner; numpy already imported |

**Key insight:** generate_tables.py is the only file that needs extension. New functions belong there — don't create a second analysis module that duplicates load/DataFrame logic.

---

## Gap Analysis: What Needs to Be Built

Based on comparing CONTEXT.md (What Needs to Be Built NEW section) against existing code:

### New Functions Required in `BackEnd/scripts/generate_tables.py`

| Function | Purpose | Decision |
|----------|---------|----------|
| `plot_convergence_curves(results_json, output_path)` | Episode reward over training time, smoothed moving average + raw faint background | D-14 |
| `select_best_config(campaign_results, metric)` | Returns config name with lowest avg_waiting_time across ablation | D-15 / D-12 |
| `plot_combined_comparison(baseline_dir, ablation_dirs, output_path)` | Side-by-side bar: baseline + best per ablation | D-15 |
| `plot_per_topology(campaign_results, topology, output_path)` | Filters df by topology, calls plot_comparison_bar | D-16 |
| `generate_ablation_table_with_pvalues(results_a, results_b, ..., output_path)` | LaTeX table with Wilcoxon p-value column | D-10/D-11 |
| `generate_combined_extensions_table(baseline_dir, ablation_dirs, output_path)` | Cross-campaign LaTeX table; highlights best values in bold | D-12 |
| `generate_experiment_report(campaign_dirs, figures_dir, output_path)` | Markdown report with Key Findings summary | D-17/D-18 |

### New CLI Script (recommended approach)

Instead of adding more flags to `generate_tables.py`, create `BackEnd/scripts/generate_report.py` as the Phase 8 entry point that:
1. Loads all four campaign directories
2. Calls all new analysis functions
3. Writes EXPERIMENT_REPORT.md
4. Can be run independently after campaigns complete

This keeps generate_tables.py as a focused per-campaign tool and generate_report.py as the cross-campaign synthesis layer.

### Topology Gap in Ablation Scripts

`run_extension_ablation.py --ablation fedprox` currently accepts only `--topology grid-3x3` (singular). Decision D-01 requires both topologies. The cleanest fix: add `--topologies` (nargs="+") as an alias matching `run_campaign.py`'s interface, with a loop over topologies inside `run_ablation()`.

Alternatively, invoke the script twice (once per topology) and use `--output-name` to save to distinct directories. But this produces split results.json files per topology rather than a combined file — which complicates per-topology chart generation (D-16).

**Recommendation:** Update `run_extension_ablation.py` to accept `--topologies` (plural) and loop internally — matches the established pattern from `run_campaign.py`.

---

## Common Pitfalls

### Pitfall 1: training_rewards=None for Baseline Configs
**What goes wrong:** Baseline campaign uses pre-loaded example weights (`weights_path` is set). `train_and_evaluate()` skips training, sets `training_rewards=None`. Attempting to build convergence curves from these results raises TypeError or produces empty charts.
**Why it happens:** `result_to_dict()` serializes `None` as JSON `null`. Reading `result["training_rewards"]` returns `None`.
**How to avoid:** Guard convergence curve generation: `if result.get("training_rewards") is not None and len(result["training_rewards"]) > 0`.
**Warning signs:** Key Findings section of report says "0 convergence curves generated."

### Pitfall 2: Wilcoxon n < 8 Returns valid=False
**What goes wrong:** `wilcoxon_compare()` returns `{"valid": False, "reason": "insufficient samples"}` when fewer than 8 seeds completed. If report generation naively reads `result["p_value"]` it will KeyError.
**Why it happens:** Any dry-run that's accidentally fed to the analysis pipeline. Also: if some MC seeds fail and `n_completed` drops below 8.
**How to avoid:** Always check `out["valid"]` before reading `out["p_value"]`. In report generator, display "n/a" for invalid tests.
**Warning signs:** KeyError on "p_value" in report generation.

### Pitfall 3: Combined Table Loads Wrong Topology Mix
**What goes wrong:** Per CONTEXT.md D-12, the combined table shows baseline vs best-FedProx vs best-cooperative vs ToD. If ablation results.json contains both topologies but the table aggregates across them, rows represent the global best rather than per-topology best, masking topology-specific scaling effects.
**Why it happens:** `results_to_dataframe()` loads all rows without topology filtering.
**How to avoid:** Filter by topology before selecting "best config". Generate two versions of the combined table — one per topology.
**Warning signs:** Combined table rows show unexpectedly low wait times that don't match either per-topology table.

### Pitfall 4: matplotlib Figure State Leaks Between Charts
**What goes wrong:** Multiple calls to `plot_comparison_bar()` or new chart functions in a single Python process accumulate figure state. Subsequent charts may include labels or artists from prior charts.
**Why it happens:** `plt.figure()` adds to global figure manager. `plt.close()` is required after each `savefig()`.
**How to avoid:** Every chart function must call `plt.close()` after `plt.savefig()`. This is already done in `plot_comparison_bar()` — mirror the pattern in all new chart functions.
**Warning signs:** Charts contain extra legend entries, axis labels from a different chart, or incorrect data.

### Pitfall 5: Ablation Campaigns Not Found
**What goes wrong:** `generate_report.py` raises `FileNotFoundError` because `results/campaigns/fedprox-ablation/results.json` doesn't exist yet.
**Why it happens:** Report generation task runs before the ablation campaign tasks complete.
**How to avoid:** Plan tasks strictly ordered: run baseline → checkpoint → run ablations → generate report. Report generation is the final task gated on all four campaign directories existing.
**Warning signs:** FileNotFoundError at report generation time.

### Pitfall 6: Single Topology Ablation (topology gap)
**What goes wrong:** Running `python scripts/run_extension_ablation.py --ablation all` only generates results for `grid-3x3` (the default). The per-topology charts (D-16) and full-scope results (D-01) are incomplete.
**Why it happens:** `run_extension_ablation.py` has `--topology` (singular, default `grid-3x3`) not `--topologies` (plural).
**How to avoid:** Either update the script to support `--topologies` before running, or invoke it explicitly for each topology. Document the invocation in plan tasks.

### Pitfall 7: Slow Wall-Clock Time
**What goes wrong:** 3 ablations x 3 configs x 2 topologies x (50 training episodes + 10 MC seeds) = up to 180 train+eval runs. On a local Windows machine, each SUMO episode may take 5-30 seconds. Total wall-clock could be 4-15 hours.
**Why it happens:** SUMO TraCI simulation is CPU-bound and not parallelised in the current implementation.
**How to avoid:** D-05 dry-run gate is the correct approach. After dry-run validates timing, project wall-clock estimate and set expectations. If too slow: reduce to grid-3x3 only for ablations, or reduce episodes (break from D-02 decision).
**Warning signs:** Dry-run 1 seed taking >2 minutes per config.

---

## Code Examples

### Training Convergence Curve (D-14)
```python
# Source: design from CONTEXT.md + existing generate_tables.py pattern
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

def plot_convergence_curves(campaign_results: list, output_path: str, window: int = 5) -> None:
    """Plot episode reward over training time for all trained configs.

    Args:
        campaign_results: List of result dicts from load_campaign_results().
        output_path: File path for the output PNG.
        window: Moving average window size for smoothing (default 5).
    """
    fig, ax = plt.subplots(figsize=(10, 5))
    plotted = 0

    for result in campaign_results:
        rewards_data = result.get("training_rewards")
        if not rewards_data:
            continue  # Skip: baseline with pre-loaded weights

        config = result.get("config", {})
        label = config.get("name", "unknown")

        # Extract per-episode mean reward — rewards_data is list of dicts
        values = [r.get("mean_reward", r) if isinstance(r, dict) else r
                  for r in rewards_data]
        episodes = list(range(1, len(values) + 1))

        # Plot raw (faint) then smoothed
        ax.plot(episodes, values, alpha=0.2, linewidth=0.8)
        if len(values) >= window:
            smoothed = np.convolve(values, np.ones(window) / window, mode="valid")
            smooth_x = list(range(window, len(values) + 1))
            ax.plot(smooth_x, smoothed, label=label, linewidth=1.8)
        plotted += 1

    if plotted == 0:
        plt.close()
        return

    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Convergence")
    ax.legend(fontsize=8)
    plt.tight_layout()
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    plt.savefig(output_path, dpi=150)
    plt.close()
```

### Best Config Selection (D-15 / D-12)
```python
def select_best_config_name(campaign_results: list, metric: str = "avg_waiting_time") -> str:
    """Return the config name with the lowest mean value of metric."""
    df = results_to_dataframe(campaign_results)
    mean_col = f"{metric}_mean"
    if df.empty or mean_col not in df.columns:
        return ""
    return df.loc[df[mean_col].idxmin(), "name"]
```

### Report Generator Skeleton (D-17/D-18)
```python
def generate_experiment_report(
    campaign_dirs: dict,   # {"baseline": path, "fedprox": path, ...}
    figures_dir: str,
    output_path: str,
) -> None:
    """Write EXPERIMENT_REPORT.md combining all tables, charts, and findings."""
    lines = ["# SEAL Experiment Report\n"]
    # Key Findings
    lines.append("## Key Findings\n")
    # ...populate from wilcoxon_compare() results...
    # Per-section references to figures
    lines.append("## Baseline Comparison\n")
    lines.append(f"![Baseline chart](figures/chart_avg_waiting_time.png)\n")
    # ...etc...
    with open(output_path, "w", encoding="utf-8") as fh:
        fh.write("\n".join(lines))
```

---

## State of the Art

| Old Approach | Current Approach | Impact |
|--------------|------------------|--------|
| Manual results transcription into LaTeX | `generate_latex_table()` auto-generates booktabs .tex from campaign JSON | All tables are reproducible with a single CLI call |
| No statistical testing | `wilcoxon_compare()` with n>=8 guard | Paper-quality significance claims |
| Visual-only bar charts | 95% CI error bars via 1.96*std/sqrt(n) | Matches standard academic figure requirements |

---

## Validation Architecture

`workflow.nyquist_validation` is not set to `false` in `.planning/config.json` — validation section included.

### Test Framework
| Property | Value |
|----------|-------|
| Framework | pytest (already installed) |
| Config file | none — invoked directly |
| Quick run command | `cd BackEnd && python -m pytest tests/test_analysis.py -x -v` |
| Full suite command | `cd BackEnd && python -m pytest tests/ -x -v --ignore=tests/test_evaluation.py` (SUMO tests skipped) |

### Phase Requirements to Test Map
| Req ID | Behavior | Test Type | Automated Command | File Exists? |
|--------|----------|-----------|-------------------|-------------|
| EXP-01 | Baseline campaign saves results.json with FedRL/MARL/SARL/fixed-time/max-pressure rows | smoke | `cd BackEnd && python scripts/run_campaign.py --campaign baseline --dry-run 1 --topologies grid-3x3` | Run-time check |
| EXP-02 | FedProx ablation saves results.json with 3 mu values | smoke | `cd BackEnd && python scripts/run_extension_ablation.py --ablation fedprox --dry-run 1 --topology grid-3x3` | Run-time check |
| EXP-03 | Cooperative ablation saves results.json with 3 alpha values | smoke | `cd BackEnd && python scripts/run_extension_ablation.py --ablation cooperative --dry-run 1 --topology grid-3x3` | Run-time check |
| EXP-04 | ToD ablation saves results.json with 2 configs | smoke | `cd BackEnd && python scripts/run_extension_ablation.py --ablation time-of-day --dry-run 1 --topology grid-3x3` | Run-time check |
| EXP-05 | plot_convergence_curves creates non-empty PNG | unit | `cd BackEnd && python -m pytest tests/test_analysis.py::test_plot_convergence_curves -x` | ❌ Wave 0 |
| EXP-05 | select_best_config_name returns lowest-wait config | unit | `cd BackEnd && python -m pytest tests/test_analysis.py::test_select_best_config -x` | ❌ Wave 0 |
| EXP-05 | generate_experiment_report creates EXPERIMENT_REPORT.md | unit | `cd BackEnd && python -m pytest tests/test_analysis.py::test_generate_report -x` | ❌ Wave 0 |
| EXP-05 | Ablation LaTeX table includes p-value column | unit | `cd BackEnd && python -m pytest tests/test_analysis.py::test_ablation_table_pvalues -x` | ❌ Wave 0 |

### Sampling Rate
- **Per task commit:** `cd BackEnd && python -m pytest tests/test_analysis.py -x -v`
- **Per wave merge:** `cd BackEnd && python -m pytest tests/ -x -v` (excluding SUMO tests)
- **Phase gate:** Full analysis test suite green + all four campaign directories populated + EXPERIMENT_REPORT.md exists

### Wave 0 Gaps
- [ ] `BackEnd/tests/test_analysis.py` — add `test_plot_convergence_curves`, `test_select_best_config`, `test_generate_report`, `test_ablation_table_pvalues` to existing test file
- [ ] No new framework install needed — pytest + matplotlib + scipy all present

---

## Open Questions

1. **Ablation topology scope — invoke twice or update script?**
   - What we know: `run_extension_ablation.py` only accepts `--topology` (singular). D-01 requires both topologies.
   - What's unclear: Whether to update the ablation script to add `--topologies` (cleaner, more complete results.json) or invoke it twice with separate output names.
   - Recommendation: Update the script to support `--topologies` (plural, nargs="+") mirroring `run_campaign.py`. This is a small change (add flag, wrap `run_ablation()` in topology loop) and produces a single combined results.json per ablation, which simplifies chart generation.

2. **training_rewards structure in results.json**
   - What we know: `CampaignResult.training_rewards = training_output.get("rewards", [])` from `run_training_loop()`. The exact shape of each element is not visible in campaign_config.py.
   - What's unclear: Whether each element is a scalar (episode total reward) or a dict (per-client rewards).
   - Recommendation: Before implementing `plot_convergence_curves`, add a Wave 0 task that prints `training_rewards[0]` structure from a dry-run result. Design the extractor to handle both shapes.

3. **Wall-clock time estimate for full campaign**
   - What we know: EXP-02 + EXP-03 + EXP-04 each train fresh weights (not pre-loaded), 50 episodes each, 10 MC seeds, 2 topologies.
   - What's unclear: How long one SUMO episode takes on this specific machine (Windows 11, local SUMO install).
   - Recommendation: Dry-run gate (D-05) measures wall-clock for 1 seed. Multiply by 10 to estimate full run cost before proceeding.

---

## Sources

### Primary (HIGH confidence)
- Direct code inspection: `BackEnd/scripts/run_campaign.py` — campaign execution interface confirmed
- Direct code inspection: `BackEnd/scripts/run_extension_ablation.py` — ablation topology gap confirmed
- Direct code inspection: `BackEnd/scripts/generate_tables.py` — existing analysis functions catalogued
- Direct code inspection: `BackEnd/api/evaluation/campaign_config.py` — CampaignResult structure confirmed
- Direct code inspection: `BackEnd/tests/test_analysis.py` — existing test patterns for new test additions

### Secondary (MEDIUM confidence)
- `.planning/phases/07-experiment-campaigns/07-01-SUMMARY.md` — campaign infrastructure decisions
- `.planning/phases/07-experiment-campaigns/07-02-SUMMARY.md` — ablation script decisions
- `.planning/phases/07-experiment-campaigns/07-03-SUMMARY.md` — analysis pipeline decisions
- `.planning/phases/08-experiment-execution-analysis/08-CONTEXT.md` — user decisions

---

## Metadata

**Confidence breakdown:**
- What exists (run_campaign.py, run_extension_ablation.py, generate_tables.py): HIGH — direct code inspection
- Topology gap in ablation script: HIGH — direct code inspection
- training_rewards element shape: LOW — not confirmed without running a dry-run
- Wall-clock time estimates: LOW — depends on hardware

**Research date:** 2026-03-23
**Valid until:** 2026-04-23 (stable codebase, no external API dependencies)
