---
phase: 08-experiment-execution-analysis
plan: 02
subsystem: experiments
tags: [sumo, fedrl, marl, sarl, fixed-time, max-pressure, campaign, monte-carlo, latex, matplotlib]

# Dependency graph
requires:
  - phase: 08-01
    provides: generate_tables.py, generate_report.py, run_campaign.py with full analysis functions

provides:
  - Baseline campaign results.json (10 configs x 10 MC seeds on grid-3x3 and grid-5x5)
  - LaTeX table: table_avg_waiting_time.tex and table_avg_travel_time.tex
  - Bar charts: chart_avg_waiting_time.png, chart_avg_travel_time.png, chart_mean_reward.png
  - User-approved baseline results gating ablation runs

affects:
  - 08-03 (ablation campaigns depend on baseline approval and results.json)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - Dry-run gate (1 seed) before full MC campaign to catch pipeline errors cheaply
    - Campaign results persisted as JSON with "results" wrapper key for generate_tables compatibility
    - User checkpoint (D-06) blocks ablation progression until baseline sanity confirmed

key-files:
  created: []
  modified:
    - BackEnd/results/campaigns/baseline/results.json
    - BackEnd/results/campaigns/baseline/config.json
    - BackEnd/results/figures/table_avg_waiting_time.tex
    - BackEnd/results/figures/table_avg_travel_time.tex
    - BackEnd/results/figures/chart_avg_waiting_time.png
    - BackEnd/results/figures/chart_avg_travel_time.png
    - BackEnd/results/figures/chart_mean_reward.png

key-decisions:
  - "Baseline campaign run with 10 MC seeds per config (D-03), 0 errors across all 10 configs"
  - "User reviewed and approved baseline avg wait times before ablation runs proceed (D-06)"
  - "results_to_dataframe unwraps campaign_to_dict serialization layer — fix applied during Task 1 to unblock generate_tables"

patterns-established:
  - "Campaign result format: {results: [...]} wrapper is mandatory for generate_tables.py compatibility"
  - "Dry-run (--dry-run 1) before full campaign is standard protocol (D-05)"

requirements-completed: [EXP-01]

# Metrics
duration: ~30min (continuation from checkpoint)
completed: 2026-03-23
---

# Phase 08 Plan 02: Baseline Campaign Execution Summary

**Baseline FedRL/MARL/SARL/fixed-time/max-pressure reproduction campaign completed — 10 configs x 10 MC seeds on grid-3x3 and grid-5x5 with 0 errors, LaTeX tables and bar charts generated, user-approved.**

## Performance

- **Duration:** ~30 min (Task 1 execution + checkpoint approval)
- **Started:** 2026-03-23 (continuation from 08-01)
- **Completed:** 2026-03-23
- **Tasks:** 2 of 2
- **Files modified:** 7 (results.json, config.json, 3 charts, 2 LaTeX tables)

## Accomplishments

- Full baseline Monte Carlo campaign completed: 10 configs (FedRL, MARL, SARL, fixed-time, max-pressure on both grid-3x3 and grid-5x5), 10 seeds each, 0 evaluation failures
- LaTeX comparison tables (table_avg_waiting_time.tex, table_avg_travel_time.tex) generated with `\toprule` formatting ready for paper inclusion
- Bar charts with 95% CI error bars generated (chart_avg_waiting_time.png, chart_avg_travel_time.png, chart_mean_reward.png)
- User reviewed and approved baseline avg wait times — gate D-06 passed, ablation runs may proceed

## Task Commits

Each task was committed atomically:

1. **Task 1: Dry-run + full baseline campaign + generate tables/charts** - `2db3044` (feat) + `1d0d618` (fix)
2. **Task 2: User reviews baseline results (checkpoint:human-verify)** - Approved by user, no code commit

## Files Created/Modified

- `BackEnd/results/campaigns/baseline/results.json` — 10 config entries, 100 MC runs total, 0 errors
- `BackEnd/results/campaigns/baseline/config.json` — Campaign configuration metadata
- `BackEnd/results/figures/table_avg_waiting_time.tex` — LaTeX baseline comparison table (D-09)
- `BackEnd/results/figures/table_avg_travel_time.tex` — LaTeX travel time table
- `BackEnd/results/figures/chart_avg_waiting_time.png` — Baseline bar chart with 95% CI (D-13)
- `BackEnd/results/figures/chart_avg_travel_time.png` — Travel time bar chart
- `BackEnd/results/figures/chart_mean_reward.png` — Mean reward bar chart

## Decisions Made

- Baseline approved by user with 0 errors across all 10 configs — no anomalies requiring investigation before ablation runs
- Fix applied to `results_to_dataframe` and `wilcoxon_compare` to unwrap the `campaign_to_dict` serialization layer (Rule 1 auto-fix during Task 1, committed in 1d0d618)

## Deviations from Plan

### Auto-fixed Issues

**1. [Rule 1 - Bug] Fixed campaign_to_dict serialization unwrap in results_to_dataframe and wilcoxon_compare**
- **Found during:** Task 1 (generate_tables step)
- **Issue:** generate_tables.py called results_to_dataframe which expected flat result dicts but received campaign_to_dict-serialized objects with nested structure
- **Fix:** Updated results_to_dataframe and wilcoxon_compare to unwrap the serialization layer before processing
- **Files modified:** BackEnd/scripts/generate_tables.py (or relevant script)
- **Verification:** All 5 figures generated successfully after fix
- **Committed in:** 1d0d618

---

**Total deviations:** 1 auto-fixed (1 bug fix)
**Impact on plan:** Fix was necessary for generate_tables.py to produce output. No scope creep.

## Issues Encountered

- generate_tables.py failed on first run due to campaign_to_dict serialization mismatch — fixed inline per deviation Rule 1, re-ran successfully

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- Baseline campaign data is approved and persisted at `BackEnd/results/campaigns/baseline/results.json`
- All 3 figure types (LaTeX tables, bar charts) are in `BackEnd/results/figures/`
- Plan 08-03 (ablation campaigns) may now proceed — dependency D-06 satisfied
- No blockers

---
*Phase: 08-experiment-execution-analysis*
*Completed: 2026-03-23*
