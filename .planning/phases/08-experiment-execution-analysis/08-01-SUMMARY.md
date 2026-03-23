---
phase: 08-experiment-execution-analysis
plan: 01
subsystem: analysis
tags: [matplotlib, pandas, scipy, latex, tdd, ablation, convergence-curves, wilcoxon]

# Dependency graph
requires:
  - phase: 07-experiment-campaigns
    provides: generate_tables.py (wilcoxon_compare, generate_latex_table, plot_comparison_bar, generate_all_outputs), run_extension_ablation.py, campaign results.json structure
provides:
  - plot_convergence_curves — training reward curves with moving-average smoothing
  - select_best_config_name — lowest-mean-metric config selector
  - plot_combined_comparison — baseline vs best-per-ablation bar chart
  - plot_per_topology — per-topology filtered bar chart
  - generate_ablation_table_with_pvalues — booktabs LaTeX with Wilcoxon p-value column
  - generate_combined_extensions_table — cross-ablation summary table with improvement %
  - generate_report.py CLI — EXPERIMENT_REPORT.md cross-campaign synthesis
  - --topologies flag on run_extension_ablation.py for multi-topology runs
affects:
  - 08-02 (baseline campaign execution — calls run_campaign.py then generate_report.py)
  - 08-03 (ablation campaign execution — calls run_extension_ablation.py then generate_report.py)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - matplotlib Agg backend with plt.close() after every plt.savefig() — prevents figure state leaks
    - moving-average smoothing via np.convolve for convergence curve readability
    - select_best_config_name as selector for cross-ablation comparison
    - booktabs LaTeX with \\textbf{} for best value highlighting

key-files:
  created:
    - BackEnd/scripts/generate_report.py — cross-campaign EXPERIMENT_REPORT.md generator CLI
  modified:
    - BackEnd/scripts/generate_tables.py — 6 new analysis functions appended
    - BackEnd/scripts/run_extension_ablation.py — --topologies flag + topology outer loop
    - BackEnd/tests/test_analysis.py — 11 new tests covering all new functions

key-decisions:
  - "plot_combined_comparison selects FedRL trainer row from baseline (not just first row)"
  - "generate_ablation_table_with_pvalues shows '---' for baseline row and 'n/a' for invalid Wilcoxon"
  - "plot_convergence_curves returns early (with plt.close) if 0 configs have training_rewards"
  - "generate_report.py uses try/except per section — missing campaign never aborts whole report"

patterns-established:
  - "plt.close() called after every plt.savefig() — matches existing generate_tables.py pattern"
  - "All new analysis functions accept campaign_results: list (raw dicts) not DataFrames"

requirements-completed: [EXP-02, EXP-03, EXP-04, EXP-05]

# Metrics
duration: 5min
completed: 2026-03-23
---

# Phase 8 Plan 1: Analysis Functions & Report Generator Summary

**6 new analysis functions (convergence curves, per-topology charts, combined extensions table, ablation tables with Wilcoxon p-values) + generate_report.py CLI + --topologies flag on ablation script, all with 17 passing unit tests**

## Performance

- **Duration:** ~5 min
- **Started:** 2026-03-23T09:03:40Z
- **Completed:** 2026-03-23T09:08:06Z
- **Tasks:** 2
- **Files modified:** 4

## Accomplishments

- Added `--topologies` (nargs=+) to `run_extension_ablation.py` with topology outer loop, enabling multi-topology runs in one CLI call (D-01 support)
- Added 6 new analysis functions to `generate_tables.py`: `plot_convergence_curves`, `select_best_config_name`, `plot_combined_comparison`, `plot_per_topology`, `generate_ablation_table_with_pvalues`, `generate_combined_extensions_table`
- Created `generate_report.py` — standalone CLI that loads all campaign dirs, calls all analysis functions, and writes a 9-section EXPERIMENT_REPORT.md (D-17, D-18)
- 17 unit tests pass; all use synthetic data (no SUMO required)

## Task Commits

1. **Task 1: --topologies flag + 6 analysis functions** - `d186f4c` (feat)
2. **Task 2: generate_report.py + report tests** - `b03e044` (feat)

## Files Created/Modified

- `BackEnd/scripts/generate_tables.py` — 6 new analysis functions: plot_convergence_curves, select_best_config_name, plot_combined_comparison, plot_per_topology, generate_ablation_table_with_pvalues, generate_combined_extensions_table
- `BackEnd/scripts/run_extension_ablation.py` — --topologies nargs=+ flag, topology outer loop in main()
- `BackEnd/scripts/generate_report.py` — NEW: cross-campaign EXPERIMENT_REPORT.md generator with 9 sections
- `BackEnd/tests/test_analysis.py` — 11 new tests (all 17 pass)

## Decisions Made

- `plot_combined_comparison` selects the FedRL trainer row from baseline results (falls back to first row if FedRL not present) — matches the paper's primary comparison target
- `generate_ablation_table_with_pvalues` shows `---` for baseline row's p-value column (no self-comparison) and `n/a` when Wilcoxon is invalid (insufficient samples or all-zero differences)
- `plot_convergence_curves` uses raw rewards as faint background (alpha=0.2, linewidth=0.8) and np.convolve moving average as solid foreground (linewidth=1.8)
- `generate_report.py` wraps each section in try/except so a missing/broken campaign never aborts the full report — missing dirs produce "not available" narrative instead

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## Next Phase Readiness

- All analysis functions ready for Plans 02 and 03 which will run actual SUMO campaigns
- `run_extension_ablation.py --topologies grid-3x3 grid-5x5` now supported
- `generate_report.py` can be called after any campaign run to get current state of all results
- Tests confirm all functions handle missing data gracefully

---
*Phase: 08-experiment-execution-analysis*
*Completed: 2026-03-23*
