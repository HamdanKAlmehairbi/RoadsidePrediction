---
phase: 03-evaluation-framework
plan: 01
subsystem: api
tags: [evaluation, fedrl, marl, sarl, sumo, tripinfo, max-pressure, ray, rllib]

# Dependency graph
requires:
  - phase: 02-wire-api
    provides: training_runner.py with get_net_file, ensure_ray, load_weights, TOPOLOGY_MAP, TRAINED_WEIGHTS_DIR
  - phase: 02-wire-api
    provides: baselines/max_pressure.py with compute_max_pressure_actions
provides:
  - BackEnd/api/evaluation/__init__.py — evaluation package
  - BackEnd/api/evaluation/runner.py — run_trial(), TrialResult, resolve_weights_path(), TRAINERS, TOPOLOGIES
  - BackEnd/api/evaluation/baselines.py — run_fixed_time_trial(), run_max_pressure_trial(), run_rl_trial()
affects:
  - 03-02 (metrics computation will call run_trial)
  - 03-03 (cross-topology transfer matrix will call run_trial)
  - 03-04 (Monte Carlo manager will call run_trial in loops)
  - 03-05 (results storage will wrap run_trial output)
  - 03-06 (API endpoints will expose run_trial via REST)

# Tech tracking
tech-stack:
  added: []
  patterns:
    - "TrialResult dataclass captures all evaluation outputs in a single return value"
    - "PPOTorchPolicy constructed standalone (no full algorithm) for inference-only evaluation"
    - "resolve_weights_path() searches trained_weights/ then example_weights/ for graceful fallback"
    - "Tripinfo XML copied to results/tripinfo/{trainer}_{topology}_{seed}.xml for persistence"

key-files:
  created:
    - BackEnd/api/evaluation/__init__.py
    - BackEnd/api/evaluation/runner.py
    - BackEnd/api/evaluation/baselines.py
  modified:
    - file-structure.md

key-decisions:
  - "run_trial() uses PPOTorchPolicy standalone (not full PPO algorithm) matching simulate.py pattern"
  - "TrialResult is a dataclass not a dict — enables type-safe access downstream"
  - "resolve_weights_path() falls back to example_weights so trials work without trained weights"
  - "max-pressure actions wrapped in try/except with fallback to fixed-time if TraCI unavailable"

patterns-established:
  - "Evaluation pattern: env_config -> SumoEnv -> episode loop -> env.close() -> parse tripinfo -> TrialResult"
  - "Weight resolution order: trained_weights/ (user's own) then example_weights/ (SEAL paper reference)"

requirements-completed:
  - EVAL-01

# Metrics
duration: 12min
completed: 2026-03-22
---

# Phase 03 Plan 01: Evaluation Runner Summary

**Single-episode evaluation runner wiring SumoEnv, PPOTorchPolicy inference, tripinfo XML parsing, and max-pressure/fixed-time baselines into a unified run_trial() API**

## Performance

- **Duration:** 12 min
- **Started:** 2026-03-22T14:36:07Z
- **Completed:** 2026-03-22T14:48:00Z
- **Tasks:** 2 of 2
- **Files modified:** 4

## Accomplishments

- Created `BackEnd/api/evaluation/runner.py` with `run_trial()` supporting all 5 trainer types across any topology in a single call
- `TrialResult` dataclass captures per-TLS rewards, tripinfo path, comm costs, step count, and vehicle count for downstream metrics computation
- `resolve_weights_path()` auto-discovers trained or example weights so trials run without manual weight path wiring
- Convenience wrappers in `baselines.py` provide clean typed entry points for fixed-time, max-pressure, and RL evaluation

## Task Commits

1. **Task 1: Create evaluation runner with run_trial()** - `5b1e7b4` (feat)
2. **Task 2: Create baseline evaluation wrappers** - `43d1dfe` (feat)

**Plan metadata:** (pending final docs commit)

## Files Created/Modified

- `BackEnd/api/evaluation/__init__.py` — Package init
- `BackEnd/api/evaluation/runner.py` — run_trial(), TrialResult, resolve_weights_path(), TRAINERS, TOPOLOGIES
- `BackEnd/api/evaluation/baselines.py` — run_fixed_time_trial(), run_max_pressure_trial(), run_rl_trial()
- `file-structure.md` — Added evaluation/ entries

## Decisions Made

**PPOTorchPolicy standalone for inference:** Mirrors the pattern already established in `simulate.py` (`_run_traci_simulation`). Avoids spinning up a full Ray algorithm object just for inference — faster and lighter for evaluation loops.

**TrialResult as dataclass:** Downstream modules (metrics, Monte Carlo, results storage) need typed field access. A plain dict would require string keys everywhere; a dataclass gives IDE completion and prevents typos.

**resolve_weights_path() fallback chain:** Researchers may not have freshly trained weights. The fallback to `example_weights/ICCPS/Final/` means all 5 trainer types work out-of-box with the reference SEAL paper weights.

**max-pressure graceful fallback:** `compute_max_pressure_actions` requires an active TraCI connection. Wrapping it in try/except with fixed-time fallback prevents evaluation from crashing when called in non-TraCI test contexts.

## Deviations from Plan

None — plan executed exactly as written.
