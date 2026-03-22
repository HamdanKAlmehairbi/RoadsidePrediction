---
phase: 03-evaluation-framework
plan: 04
subsystem: api
tags: [fastapi, websocket, asyncio, json-persistence, monte-carlo, evaluation]

# Dependency graph
requires:
  - phase: 03-03
    provides: run_full_campaign, campaign_to_dict, MCConfig, MCAggregatedResult
  - phase: 03-01
    provides: runner.py, TRAINERS, TOPOLOGIES
  - phase: 03-02
    provides: transfer.py, build_transfer_matrix, transfer_matrix_to_dict

provides:
  - POST /api/evaluate (202) — starts async evaluation campaign, returns job_id
  - GET /api/evaluation/{job_id} — returns full aggregated results
  - GET /api/evaluations — lists all stored evaluations (summary metadata)
  - WS /ws/evaluate/{job_id} — streams per-trial progress frames
  - BackEnd/results/evaluations/{job_id}.json — persistent evaluation storage

affects: [03-05, UI wiring for evaluation page, frontend evaluation hooks]

# Tech tracking
tech-stack:
  added: []
  patterns:
    - asyncio.create_task -> ThreadPoolExecutor -> sync worker (same as train.py)
    - on_progress callback thread-safely puts frames to asyncio.Queue
    - dual-path result lookup (in-memory fast path, disk slow path for restarts)
    - json.dump with default=str for safe serialization of arbitrary result types

key-files:
  created:
    - BackEnd/api/evaluation/store.py
    - BackEnd/api/routes/evaluate.py
    - BackEnd/api/ws/evaluate.py
  modified:
    - BackEnd/api/main.py

key-decisions:
  - "dual-path result lookup: check in-memory jobs dict first, fall back to disk load"
  - "json.dump(default=str) in store.py for safety with non-serializable dataclass fields"
  - "error results also persisted to disk so GET /api/evaluation/{job_id} always returns something"
  - "WS closes on both 'complete' and 'error' job status to prevent client hanging"
  - "include_transfer defaults False — transfer matrix is expensive, opt-in only"

patterns-established:
  - "Evaluation API pattern: POST /api/evaluate -> job_id -> WS for progress -> GET for results"
  - "Persistent results: save_evaluation() after every job (success or error), load_evaluation() on demand"

requirements-completed: [EVAL-06, EVAL-07, EVAL-08, EVAL-09]

# Metrics
duration: 8min
completed: 2026-03-22
---

# Phase 03 Plan 04: Evaluation API and Storage Summary

**Persistent JSON evaluation store + REST API (POST/GET) + WebSocket streaming wired to Monte Carlo campaign runner.**

## Performance

- **Duration:** ~8 min
- **Started:** 2026-03-22T16:09Z
- **Completed:** 2026-03-22T16:17Z
- **Tasks:** 2/2
- **Files modified:** 4

## Accomplishments

- Created `store.py` with `RESULTS_DIR` auto-init, save/load/list/delete operations — evaluations survive server restart
- Created `routes/evaluate.py` with POST /api/evaluate, GET /api/evaluation/{job_id}, GET /api/evaluations — follows exact train.py async pattern
- Created `ws/evaluate.py` streaming per-trial progress frames via asyncio queue — closes cleanly on complete or error
- Registered both new routers in main.py — full app imports without errors
- Optional cross-topology transfer matrix supported via `include_transfer=True` on request

## Task Commits

1. **Task 1: Create persistent results store** - `8eb0c75` (feat)
2. **Task 2: Create evaluation API endpoints and WebSocket** - `ac9f858` (feat)

## Files Created/Modified

- `BackEnd/api/evaluation/store.py` — JSON persistence: RESULTS_DIR, save/load/list/delete evaluation
- `BackEnd/api/routes/evaluate.py` — REST endpoints: POST /api/evaluate, GET /api/evaluation/{job_id}, GET /api/evaluations
- `BackEnd/api/ws/evaluate.py` — WebSocket: /ws/evaluate/{job_id} streams progress frames
- `BackEnd/api/main.py` — Added evaluate_router and ws_evaluate_router includes

## Decisions Made

- **Dual-path result lookup**: GET /api/evaluation/{job_id} checks in-memory `jobs` dict first (fast for recently-completed), then falls back to disk load (handles server-restart scenario). This avoids forcing clients to re-run evaluations after a server restart.
- **json.dump with default=str**: store.py uses this to safely handle any non-serializable types that might appear in the results dict (datetime, dataclass remnants, numpy scalars from runner.py).
- **Error results persisted**: If the evaluation job fails, an error dict is still saved to disk so GET /api/evaluation/{job_id} always returns a useful response with status "error".
- **WS closes on both "complete" and "error"**: Prevents WebSocket clients from hanging if the campaign throws an exception.
- **include_transfer defaults False**: Transfer matrix is computationally expensive (len(TOPOLOGIES)^2 trials per RL trainer). Opt-in only.

## Deviations from Plan

None — plan executed exactly as written.

## Next Phase Readiness

- All EVAL-06 through EVAL-09 requirements met
- Evaluation endpoints ready for frontend wiring (Phase 03-05 and UI phase)
- Results persisted in `BackEnd/results/evaluations/` — frontend can list and fetch by job_id
- Transfer matrix endpoint available via `include_transfer: true` in request body

---
*Phase: 03-evaluation-framework*
*Completed: 2026-03-22*
