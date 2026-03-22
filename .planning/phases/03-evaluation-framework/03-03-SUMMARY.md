---
phase: 03-evaluation-framework
plan: "03"
subsystem: evaluation
tags: [monte-carlo, statistics, confidence-interval, seed-control, aggregation, python, fedrl]

# Dependency graph
requires:
  - phase: 03-01
    provides: run_trial(), TrialResult, resolve_weights_path(), TRAINERS, TOPOLOGIES
  - phase: 03-02
    provides: compute_trial_metrics(), TrialMetrics, metrics_to_dict()

provides:
  - MCConfig dataclass for reproducible N-run configurations
  - MCAggregatedResult dataclass with per-run and statistical summaries
  - run_monte_carlo() — N trials with base_seed+i seeding and progress callback
  - run_full_campaign() — 5 trainers x 3 topologies x 10 runs (150 trials)
  - campaign_to_dict() — JSON-serializable campaign output
  - _aggregate_metrics() — mean, std, 95% CI, min, max (stdlib only, no numpy)

affects:
  - 03-04 (storage/persistence — consumes MCAggregatedResult and campaign_to_dict output)
  - 03-05 (API endpoints — wraps run_full_campaign for async jobs)
  - UI-03 (Evaluation page results tables)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "Monte Carlo seeding: base_seed + i for reproducible multi-run experiments"
    - "Graceful failure isolation: individual run exceptions logged but don't abort campaign"
    - "Progress callback pattern: on_progress(completed, total, config) for streaming"
    - "Stdlib-only aggregation: statistics.mean/stdev + math.sqrt (no numpy in this module)"

key-files:
  created:
    - BackEnd/api/evaluation/monte_carlo.py
  modified:
    - file-structure.md

key-decisions:
  - "Use stdlib statistics+math for aggregation — avoids numpy dependency in orchestration layer"
  - "Resolve weights once outside N-run loop — single FileNotFoundError aborts config, not run-by-run"
  - "on_progress callback swallows its own exceptions — user callback errors never abort the campaign"
  - "n=1 run gives std=0 and CI = mean +/- 0 (degenerate but valid)"

patterns-established:
  - "MCConfig: single source of truth for all trial parameters, passed by value through the stack"
  - "MCAggregatedResult: separates individual_results (serializable) from successful_metrics (internal)"

requirements-completed:
  - EVAL-05

# Metrics
duration: 2min
completed: 2026-03-22
---

# Phase 03 Plan 03: Monte Carlo Orchestration Summary

**N-run Monte Carlo campaign orchestrator with 95% CI aggregation: run_full_campaign() drives 150 trials (5 trainers x 3 topologies x 10 seeds) using stdlib statistics, seed=base_seed+i control, and graceful per-run failure isolation.**

## Performance

- **Duration:** 2 min
- **Started:** 2026-03-22T16:06:06Z
- **Completed:** 2026-03-22T16:08:02Z
- **Tasks:** 1
- **Files modified:** 2 (monte_carlo.py created, file-structure.md updated)

## Accomplishments
- Created MCConfig dataclass with all reproducibility fields (trainer, topology, n_runs=10, base_seed=42, ranked, horizon, weights_path)
- Created MCAggregatedResult capturing per-run dicts, aggregate stats, completion counts, and error messages
- Implemented _aggregate_metrics() returning mean/std/ci_95_lower/ci_95_upper/min/max for 5 key metrics using pure stdlib
- Implemented run_monte_carlo() with seed=base_seed+i loop, on_progress callback, and graceful exception handling
- Implemented run_full_campaign() iterating the full 5x3 grid with configurable subsets
- Implemented campaign_to_dict() producing a fully JSON-serializable nested structure

## Task Commits

Each task was committed atomically:

1. **Task 1: Create Monte Carlo orchestration module** - `c8515ec` (feat)

**Plan metadata:** (this SUMMARY commit)

## Files Created/Modified
- `BackEnd/api/evaluation/monte_carlo.py` — MCConfig, MCAggregatedResult, run_monte_carlo, run_full_campaign, campaign_to_dict, _aggregate_metrics
- `file-structure.md` — Added evaluation subdirectory entries for metrics.py, transfer.py, monte_carlo.py

## Decisions Made
- **Stdlib-only**: Used `statistics.mean`, `statistics.stdev`, and `math.sqrt` instead of numpy — keeps aggregation layer free of heavy scientific stack dependency; numpy already used in runner.py for policy inference but not needed here
- **Weights resolved once**: `resolve_weights_path()` called once before the N-run loop; if weights are missing the entire config fails fast with a clear error rather than failing silently on each run
- **Callback exception swallowed**: `on_progress` exceptions are caught with DEBUG logging — a broken UI callback should never abort a long-running campaign
- **Single n=1 degenerate case**: `statistics.stdev` requires n>1; guarded with `if n > 1 else 0.0` to allow single-run smoke tests

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None - no external service configuration required.

## Next Phase Readiness
- monte_carlo.py is ready for consumption by 03-04 (storage) and 03-05 (API endpoints)
- campaign_to_dict() output is directly usable by FastAPI response models
- on_progress callback signature (completed, total, config) is designed for WebSocket streaming
- No blockers identified

---
*Phase: 03-evaluation-framework*
*Completed: 2026-03-22*
