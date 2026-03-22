---
phase: 03-evaluation-framework
verified: 2026-03-22T16:20:18Z
status: passed
score: 11/11 must-haves verified
gaps: []
---

# Phase 3: Evaluation Framework Verification Report

**Phase Goal:** Automated evaluation campaigns producing publishable results tables
**Verified:** 2026-03-22T16:20:18Z
**Status:** passed
**Re-verification:** No - initial verification

---

## Goal Achievement

### Observable Truths

| # | Truth | Status | Evidence |
|---|-------|--------|----------|
| 1 | run_trial() executes on any topology/trainer | VERIFIED | runner.py:127 all 5 trainers via branching; episode loop runs to completion |
| 2 | Fixed-time and max-pressure baselines produce valid metrics | VERIFIED | baselines.py:23,53 delegate to run_trial; max-pressure falls back if TraCI unavailable |
| 3 | RL evaluation loads trained weights and runs inference | VERIFIED | runner.py:172-197 PPOTorchPolicy standalone, weights resolved, compute_single_action |
| 4 | Each trial returns per-TLS rewards, tripinfo path, and comm cost dict | VERIFIED | TrialResult runner.py:56 has per_tls_rewards, tripinfo_path, comm_costs; all populated |
| 5 | Metrics: avg waiting time, avg travel time, throughput from tripinfo XML | VERIFIED | metrics.py:86 ET.parse extracts waitingTime/duration/departDelay; test validates |
| 6 | Metrics: mean episode reward, communication cost by type | VERIFIED | metrics.py:141 compute_trial_metrics flattens per_tls_rewards, sums comm_costs |
| 7 | Transfer matrix built by running train-on-X test-on-Y for all pairs | VERIFIED | transfer.py:56 nested TOPOLOGIES x TOPOLOGIES loop; skips missing weights |
| 8 | Transfer gap metric quantifies performance degradation | VERIFIED | transfer.py:131 gap=home_perf-transfer_perf; gap_pct; test validates gap=1.0 |
| 9 | Monte Carlo runs N trials with different seeds and 95% CI | VERIFIED | monte_carlo.py:153 seed=base_seed+i; CI: 1.96*std/sqrt(n) confirmed |
| 10 | Full 5 trainers x 3 topologies x 10 MC runs campaign executes | VERIFIED | monte_carlo.py:246 run_full_campaign iterates TRAINERS x TOPOLOGIES; 150 trials |
| 11 | Results persist as JSON and queryable via POST/WS/GET endpoints | VERIFIED | store.py + routes/evaluate.py + ws/evaluate.py all wired in main.py |

**Score:** 11/11 truths verified

---

### Required Artifacts

| Artifact | Lines | Status |
|----------|-------|--------|
| BackEnd/api/evaluation/__init__.py | 33 | VERIFIED: re-exports 22 symbols, __all__ defined |
| BackEnd/api/evaluation/runner.py | 308 | VERIFIED: run_trial, TrialResult, resolve_weights_path, TRAINERS, TOPOLOGIES |
| BackEnd/api/evaluation/baselines.py | 130 | VERIFIED: run_fixed_time_trial, run_max_pressure_trial, run_rl_trial |
| BackEnd/api/evaluation/metrics.py | 211 | VERIFIED: TripinfoMetrics, TrialMetrics, compute_tripinfo_metrics, compute_trial_metrics, metrics_to_dict |
| BackEnd/api/evaluation/transfer.py | 224 | VERIFIED: TransferResult, build_transfer_matrix, compute_transfer_gap, transfer_matrix_to_dict |
| BackEnd/api/evaluation/monte_carlo.py | 346 | VERIFIED: MCConfig, MCAggregatedResult, run_monte_carlo, run_full_campaign, campaign_to_dict |
| BackEnd/api/evaluation/store.py | 129 | VERIFIED: RESULTS_DIR, save_evaluation, load_evaluation, list_evaluations, delete_evaluation |
| BackEnd/api/routes/evaluate.py | 233 | VERIFIED: POST /api/evaluate (202), GET /api/evaluation/{job_id}, GET /api/evaluations |
| BackEnd/api/ws/evaluate.py | 44 | VERIFIED: WS /ws/evaluate/{job_id}; reads asyncio queue; closes on complete/error |
| BackEnd/tests/test_evaluation.py | 310 | VERIFIED: 11 tests covering all EVAL requirements |

---

### Key Link Verification

| From | To | Via | Status |
|------|----|-----|--------|
| runner.py | training_runner.py | from ..training_runner import get_net_file, ensure_ray, load_weights, TOPOLOGY_MAP | WIRED |
| runner.py | seal.sumo.env.SumoEnv | from seal.sumo.env import SumoEnv (lazy import in run_trial) | WIRED |
| runner.py | tripinfo XML | ET.parse(TRIPINFO_OUT_FILENAME) after env.close() | WIRED |
| metrics.py | tripinfo XML files | ET.parse(tripinfo_path) in compute_tripinfo_metrics | WIRED |
| transfer.py | runner.py | from .runner import run_trial, resolve_weights_path, TOPOLOGIES | WIRED |
| transfer.py | metrics.py | from .metrics import TrialMetrics, compute_trial_metrics, metrics_to_dict | WIRED |
| monte_carlo.py | runner.py | from .runner import run_trial, resolve_weights_path, TRAINERS, TOPOLOGIES | WIRED |
| monte_carlo.py | metrics.py | from .metrics import compute_trial_metrics, metrics_to_dict, TrialMetrics | WIRED |
| routes/evaluate.py | monte_carlo.py | from ..evaluation.monte_carlo import run_full_campaign, campaign_to_dict | WIRED |
| routes/evaluate.py | store.py | from ..evaluation.store import save_evaluation, load_evaluation, list_evaluations | WIRED |
| routes/evaluate.py | jobs.py | from ..jobs import jobs, create_job | WIRED |
| ws/evaluate.py | jobs.py | from ..jobs import jobs (reads frames_queue) | WIRED |
| main.py | routes/evaluate.py | from .routes.evaluate import router as evaluate_router + app.include_router | WIRED |
| main.py | ws/evaluate.py | from .ws.evaluate import router as ws_evaluate_router + app.include_router | WIRED |

---

### Requirements Coverage

| Requirement | Description | Status | Supporting Artifact |
|-------------|-------------|--------|---------------------|
| EVAL-01 | Evaluation runner orchestrates campaigns across trainers x topologies | SATISFIED | runner.py:run_trial + monte_carlo.py:run_full_campaign |
| EVAL-02 | Metrics: avg waiting time, avg travel time, throughput from tripinfo | SATISFIED | metrics.py:compute_tripinfo_metrics |
| EVAL-03 | Metrics: mean episode reward, communication cost by type | SATISFIED | metrics.py:compute_trial_metrics |
| EVAL-04 | Cross-topology transfer matrix (train on X, test on Y) | SATISFIED | transfer.py:build_transfer_matrix |
| EVAL-05 | Monte Carlo runs with seed control, 10 per config | SATISFIED | monte_carlo.py:run_monte_carlo (base_seed+i seeding, n_runs=10 default) |
| EVAL-06 | Persistent results in BackEnd/results/ as JSON | SATISFIED | store.py RESULTS_DIR = BackEnd/results/evaluations |
| EVAL-07 | POST /api/evaluate starts evaluation job | SATISFIED | routes/evaluate.py @router.post status_code=202 |
| EVAL-08 | WS /ws/evaluate/{job_id} streams per-trial results | SATISFIED | ws/evaluate.py @router.websocket |
| EVAL-09 | GET /api/evaluation/{job_id} returns aggregated results | SATISFIED | routes/evaluate.py dual-path lookup (in-memory + disk) |
| EVAL-10 | Fixed-time and max-pressure baselines included in evaluation | SATISFIED | baselines.py; TRAINERS constant includes both |
| EVAL-11 | Transfer gap metric computed across topologies | SATISFIED | transfer.py:compute_transfer_gap returns gap and gap_pct per trainer |

**All 11 EVAL requirements: SATISFIED**

---

### Anti-Patterns Found

| File | Line | Pattern | Severity | Impact |
|------|------|---------|----------|--------|
| monte_carlo.py | 116 | return {} | Info | Legitimate guard clause when metrics_list is empty; not a stub |

No blockers or warnings. No TODO/FIXME/placeholder patterns found in any evaluation module.

---

### Human Verification Required

### 1. Full campaign runtime

**Test:** POST /api/evaluate with default body (5 trainers x 3 topologies x 10 runs)
**Expected:** Completes; GET /api/evaluation/{job_id} returns table with mean and std for all metrics
**Why human:** Requires SUMO running and trained weights; 150 SUMO episodes take significant wall time

### 2. FedRL vs SARL communication cost difference

**Test:** Run campaign and compare total_comm_cost mean for FedRL vs SARL in results
**Expected:** FedRL shows EDGE2TLS_POLICY messages; measurably different comm profile from SARL
**Why human:** Requires trained FedRL weights and live SUMO to generate real comm events

### 3. Transfer gap measurability

**Test:** Run build_transfer_matrix for FedRL with trained weights for all 3 topologies
**Expected:** gap_pct > 0 for off-diagonal entries showing performance degradation
**Why human:** Gap magnitude is empirical; requires weights for all 3 topologies

### 4. Results persistence across server restart

**Test:** Complete an evaluation, restart BackEnd server, call GET /api/evaluation/{job_id}
**Expected:** Full results returned from disk via load_evaluation fallback path
**Why human:** Requires actually restarting the server process

---

## Gaps Summary

None. All 11 EVAL requirements are structurally satisfied. Every required artifact exists,
is substantive (33-346 lines of real implementation), and is wired correctly to the rest
of the system. The evaluation framework is complete as a codebase artifact.

---

*Verified: 2026-03-22T16:20:18Z*
*Verifier: Claude (gsd-verifier)*
