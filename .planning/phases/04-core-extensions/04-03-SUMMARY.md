---
phase: 04-core-extensions
plan: 03
subsystem: rl-environment
tags: [time-of-day, demand-curriculum, time-encoding, sine-cosine, api-wiring, phase4-params]

# Dependency graph
requires:
  - phase: 04-core-extensions
    provides: "FedProx fedprox_mu in FedPolicyTrainer (04-01), cooperative alpha in SumoEnv (04-02)"
  - phase: 02-wire-api
    provides: "Training runner, evaluation runner, BaseTrainer env_config_fn"
provides:
  - "Time-of-day demand curriculum in AbstractSumoEnv (am_rush/midday/pm_rush)"
  - "Sine/cosine time encoding in SumoEnv observations"
  - "Full API wiring of all Phase 4 params: fedprox_mu, alpha, time_of_day, use_time_encoding"
  - "TrainRequest with four new Phase 4 extension fields"
affects: [experiment-campaigns, evaluation-runs, frontend-training-ui]

# Tech tracking
tech-stack:
  added: []
  patterns: ["per-episode demand sampling for curriculum training", "sine/cosine temporal encoding appended after ranking"]

key-files:
  created: []
  modified:
    - BackEnd/seal/sumo/abstract_env.py
    - BackEnd/seal/sumo/env.py
    - BackEnd/api/routes/train.py
    - BackEnd/api/training_runner.py
    - BackEnd/seal/trainer/base.py
    - BackEnd/api/evaluation/runner.py

key-decisions:
  - "Time encoding appended AFTER ranking to preserve rank indices 10-13 intact"
  - "Per-episode demand sampling (not intra-episode) for training diversity without complexity"
  - "time_of_day is training-only; evaluation uses fixed seeds for reproducibility"

patterns-established:
  - "Demand curriculum: random.choice of period profiles per episode reset"
  - "Temporal encoding: sin/cos of (2*pi*step/horizon) appended to obs vector"
  - "Full param passthrough: TrainRequest -> create_trainer -> BaseTrainer.env_config_fn -> SumoEnv"

requirements-completed: [EXT-03, EXT-04]

# Metrics
duration: 3min
completed: 2026-03-23
---

# Phase 04 Plan 03: Time-of-Day Demand and API Wiring Summary

**Time-of-day demand curriculum with sin/cos time encoding and full Phase 4 parameter wiring through API to environment**

## Performance

- **Duration:** 3 min
- **Started:** 2026-03-23T03:30:45Z
- **Completed:** 2026-03-23T03:34:03Z
- **Tasks:** 2
- **Files modified:** 6

## Accomplishments
- Time-of-day demand curriculum varies vplph per episode across three profiles (am_rush 500-700, midday 200-300, pm_rush 400-600)
- Sine/cosine time encoding appends 2 features to observation vector, observation_space grows from Box(14) to Box(16)
- All four Phase 4 parameters (fedprox_mu, alpha, time_of_day, use_time_encoding) flow end-to-end from TrainRequest through create_trainer, BaseTrainer.env_config_fn, into SumoEnv config
- build_inference_algorithm accepts use_time_encoding for correct obs space at inference time
- run_trial accepts alpha and use_time_encoding for evaluation with cooperative reward

## Task Commits

Each task was committed atomically:

1. **Task 1: Add time-of-day demand and time encoding to environment layer** - `f084337` (feat)
2. **Task 2: Wire all Phase 4 params through API layer** - `cdfbfc0` (feat)

## Files Created/Modified
- `BackEnd/seal/sumo/abstract_env.py` - time_of_day flag, forced rand_routes_on_reset, demand variation in rand_routes()
- `BackEnd/seal/sumo/env.py` - use_time_encoding, sin/cos append in _observe(), observation_space override
- `BackEnd/api/routes/train.py` - TrainRequest with 4 new fields, passthrough in all call chain functions
- `BackEnd/api/training_runner.py` - create_trainer accepts all params, common_kwargs includes env extras, build_inference_algorithm accepts use_time_encoding
- `BackEnd/seal/trainer/base.py` - kwargs reads for alpha/time_of_day/use_time_encoding, env_config_fn returns them
- `BackEnd/api/evaluation/runner.py` - run_trial accepts alpha and use_time_encoding, adds to env_config

## Decisions Made
- Time encoding appended AFTER ranking in _observe(): ranking writes at indices 10-13, sin/cos go to 14-15 so no index collision
- Per-episode demand sampling (one period per reset) rather than intra-episode variation: simpler, each episode is a consistent scenario
- time_of_day NOT passed to run_trial: evaluation uses fixed seeds for reproducibility, demand curriculum is training-only
- build_inference_algorithm gets use_time_encoding but not alpha/time_of_day: inference needs correct obs shape but not reward shaping or demand variation

## Deviations from Plan

None - plan executed exactly as written.

## Issues Encountered
None

## User Setup Required
None - no external service configuration required.

## Known Stubs
None - all functionality fully wired.

## Next Phase Readiness
- All Phase 4 extensions now selectable via API: fedprox_mu, alpha, time_of_day, use_time_encoding
- Ready for experiment campaigns comparing FedAvg vs FedProx, selfish vs cooperative, fixed vs time-varying demand
- Frontend training UI can expose the new parameters when UI phase executes

## Self-Check: PASSED

All 6 modified files verified on disk. Both task commits (f084337, cdfbfc0) confirmed in git log.

---
*Phase: 04-core-extensions*
*Completed: 2026-03-23*
