---
phase: 07-experiment-campaigns
plan: 03
subsystem: backend-evaluation
tags: [analysis, statistics, wilcoxon, latex, charts, publication]
dependency_graph:
  requires:
    - api/evaluation/campaign_config.py (result structure)
    - results/campaigns/*/results.json (campaign output)
    - scipy.stats (Wilcoxon signed-rank test)
    - matplotlib (bar chart generation)
    - pandas (DataFrame operations)
  provides:
    - generate_tables.py CLI (tables, charts, statistical tests)
    - wilcoxon_compare() — paired Wilcoxon signed-rank test
    - generate_latex_table() — booktabs-formatted LaTeX tables
    - plot_comparison_bar() — grouped bar charts with 95% CI error bars
    - results_to_dataframe() — campaign JSON to pandas DataFrame
    - generate_all_outputs() — batch generation of all artifacts
  affects:
    - Publication figures and tables
    - Research paper statistical claims
tech_stack:
  added: []
  patterns:
    - matplotlib.use("Agg") before pyplot import for headless rendering
    - Hand-crafted LaTeX with booktabs (not pandas .to_latex())
    - 95% CI as 1.96 * std / sqrt(n) for error bars
    - Wilcoxon n >= 8 guard for statistical validity
key_files:
  created:
    - BackEnd/scripts/generate_tables.py
    - BackEnd/tests/test_analysis.py
  modified: []
decisions:
  - "Hand-crafted LaTeX with booktabs rather than pandas .to_latex() — full control over formatting"
  - "Wilcoxon requires n >= 8 per condition — returns valid=False with reason for smaller samples"
  - "matplotlib Agg backend set globally before any pyplot import — prevents display issues in CI/headless"
  - "95% CI error bars use 1.96 * std / sqrt(n) — standard normal approximation"
metrics:
  completed_date: "2026-03-23T04:45:00Z"
  tasks_completed: 2
  files_created: 2
  tests_passed: 7
---

# Phase 07 Plan 03: Statistical Analysis and Publication Output Summary

**One-liner:** Publication-ready analysis pipeline with Wilcoxon significance tests, booktabs LaTeX tables, and grouped bar charts with 95% CI error bars.

## What Was Built

### BackEnd/scripts/generate_tables.py
- `load_campaign_results()`: loads results.json from campaign directory
- `results_to_dataframe()`: flattens campaign results into pandas DataFrame with metric mean/std columns
- `wilcoxon_compare()`: paired Wilcoxon signed-rank test with n >= 8 guard and graceful error handling
- `generate_latex_table()`: booktabs LaTeX with `\toprule`/`\midrule`/`\bottomrule`, mean +/- std formatting
- `plot_comparison_bar()`: grouped bar chart with 95% CI error bars, matplotlib Agg backend
- `generate_all_outputs()`: batch generation of LaTeX tables (wait time, travel time) and bar charts (3 metrics)
- argparse CLI: `--campaign-dir`, `--output-dir`, `--metric`, `--format {all,table,chart}`

### BackEnd/tests/test_analysis.py
7 unit tests using synthetic data (no SUMO required):
- `test_results_to_dataframe_columns` — verifies row count and expected columns
- `test_results_to_dataframe_skips_errors` — verifies error results are filtered
- `test_wilcoxon_compare_significant` — detects significant difference (p < 0.05)
- `test_wilcoxon_compare_insufficient_samples` — returns valid=False for n < 8
- `test_generate_latex_table_has_booktabs` — verifies toprule/midrule/bottomrule and pm formatting
- `test_plot_comparison_bar_creates_file` — verifies non-empty PNG created

## Verification Results

```
7 passed — all analysis functions work with synthetic data
scipy.stats.wilcoxon confirmed as statistical test
booktabs formatting verified in LaTeX output
matplotlib Agg backend confirmed
```

## Deviations from Plan

None — plan executed exactly as written.

## Known Stubs

None — all functions are fully implemented. They operate on campaign results.json files produced by run_campaign.py or run_extension_ablation.py.

## Self-Check: PASSED
