---
phase: 03-evaluation-framework
plan: "02"
subsystem: evaluation
tags: [sumo, tripinfo, xml, metrics, transfer, fedrl, numpy]

requires:
  - phase: 03-01
    provides: "TrialResult dataclass, run_trial(), resolve_weights_path(), TOPOLOGIES from runner.py"

provides:
  - "TripinfoMetrics dataclass: avg_waiting_time, avg_travel_time, avg_depart_delay, throughput, completed_trips"
  - "TrialMetrics dataclass: tripinfo KPIs + mean_reward, comm_costs, total_comm_cost, trainer, topology, seed"
  - "compute_tripinfo_metrics(): ET.parse SUMO tripinfo XML to extract per-vehicle KPIs"
  - "compute_trial_metrics(): combines tripinfo + per_tls_rewards + comm_costs into TrialMetrics"
  - "metrics_to_dict(): JSON-serializable conversion with 4dp float rounding"
  - "TransferResult dataclass: train_topology, test_topology, trainer, metrics, is_transfer"
  - "build_transfer_matrix(): runs all TOPOLOGIES^2 pairs, resolves weights, returns List[TransferResult]"
  - "compute_transfer_gap(): home vs transfer mean_reward gap and gap_pct by trainer"
  - "transfer_matrix_to_dict(): nested JSON structure {trainer, matrix, transfer_gap}"

affects:
  - "03-03 (Monte Carlo runner) — uses compute_trial_metrics and TrialMetrics"
  - "03-04 (results storage) — uses metrics_to_dict and transfer_matrix_to_dict"
  - "03-05 (API endpoints) — serves metrics and transfer matrix via REST"

tech-stack:
  added: []
  patterns:
    - "Dataclass-first metrics: typed dataclasses (TripinfoMetrics, TrialMetrics, TransferResult) for downstream safety"
    - "Lazy TrialResult import in compute_trial_metrics to avoid circular dependencies at module load"
    - "Graceful degradation: compute_tripinfo_metrics returns empty TripinfoMetrics on missing/malformed file"
    - "Transfer gap as home-minus-transfer: positive gap = performance degradation, gap_pct for publishable tables"

key-files:
  created:
    - BackEnd/api/evaluation/metrics.py
    - BackEnd/api/evaluation/transfer.py
  modified: []

key-decisions:
  - "Throughput defaults to 1.0 when spawned vehicle count unavailable (tripinfo only records completed trips); MC aggregation layer can override"
  - "compute_transfer_gap returns List[dict] (not single dict) to support multi-trainer comparison in one call"
  - "transfer_matrix_to_dict uses results[0].trainer as canonical trainer label — matrix is per-trainer by construction"
  - "build_transfer_matrix skips pairs with FileNotFoundError on weights (warning, not raise) for partial-results resilience"

patterns-established:
  - "XML metric extraction: ET.parse + root.findall('tripinfo') pattern (from SUMO-FedRL-main/eval.py)"
  - "Transfer matrix shape: {train_topo: {test_topo: metrics_dict}} — matches publishable table layout"
  - "Gap metric formula: gap = home - transfer, gap_pct = gap / |home| * 100"

requirements-completed:
  - EVAL-02
  - EVAL-03
  - EVAL-04
  - EVAL-11

duration: 2min
completed: 2026-03-22
---

# Phase 03 Plan 02: Metrics and Transfer Testing Summary

**TripinfoMetrics + TrialMetrics from SUMO XML, and a 3x3 cross-topology transfer matrix with home/transfer gap computation for publishable FedRL results**

## Performance

- **Duration:** ~2 min
- **Started:** 2026-03-22T14:40:09Z
- **Completed:** 2026-03-22T14:41:51Z
- **Tasks:** 2/2
- **Files modified:** 2

## Accomplishments

- `metrics.py` — parses SUMO tripinfo XML (waitingTime, duration, departDelay) into typed TripinfoMetrics; aggregates per-TLS reward lists into TrialMetrics with comm cost totals; JSON-serializes with 4dp float rounding
- `transfer.py` — builds the full TOPOLOGIES x TOPOLOGIES transfer matrix per trainer, skipping missing weights gracefully; computes home vs transfer performance gap (absolute and percentage) grouped by trainer
- Both modules import cleanly and are verified against package-level `from api.evaluation import metrics, transfer`

## Task Commits

1. **Task 1: Create metrics computation module** - `b6f09d3` (feat)
2. **Task 2: Create cross-topology transfer testing module** - `cdf83f2` (feat)

**Plan metadata:** (next commit — docs)

## Files Created/Modified

- `BackEnd/api/evaluation/metrics.py` — TripinfoMetrics, TrialMetrics, compute_tripinfo_metrics, compute_trial_metrics, metrics_to_dict
- `BackEnd/api/evaluation/transfer.py` — TransferResult, build_transfer_matrix, compute_transfer_gap, transfer_matrix_to_dict

## Decisions Made

- Throughput defaults to 1.0 in `compute_tripinfo_metrics` because the tripinfo XML only records completed trips; spawned count comes from route files which are not passed here. MC aggregation in plan 03-03 can set a real ratio.
- `compute_transfer_gap` returns `List[dict]` rather than a single dict so it handles multi-trainer result lists naturally (future: compare FedRL vs MARL vs SARL gaps in one call).
- `build_transfer_matrix` logs a warning and skips (not raises) when weights are missing for a train_topo/trainer pair, allowing partial matrices when only some topologies are trained.

## Deviations from Plan

None — plan executed exactly as written.

## Issues Encountered

None.

## User Setup Required

None — no external service configuration required.

## Next Phase Readiness

- `metrics.py` and `transfer.py` are ready for plan 03-03 (Monte Carlo run manager) to import and aggregate
- `transfer_matrix_to_dict` produces the nested structure expected by plan 03-04 (results storage) and 03-05 (API endpoints)
- Concern: `build_transfer_matrix` triggers real SUMO episodes — ensure async job pattern is in place before calling from API (addressed in 03-05)

---
*Phase: 03-evaluation-framework*
*Completed: 2026-03-22*
