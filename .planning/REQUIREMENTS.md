# Requirements: SEAL Dashboard

**Defined:** 2026-03-22
**Core Value:** Real FedRL experiments working end-to-end with publishable results

## v1 Requirements

### Environment Setup (ENV)

- [x] **ENV-01**: SUMO + TraCI installed and working with Python bindings
- [x] **ENV-02**: Ray 2.x, gymnasium, PyTorch 2.x environment configured
- [x] **ENV-03**: SEAL engine imports succeed (`SumoEnv`, `FedPolicyTrainer`)

### Training Pipeline (TRAIN)

- [x] **TRAIN-01**: FedRL training runs end-to-end on grid-3x3 with real SUMO
- [x] **TRAIN-02**: MARL and SARL training work as alternatives
- [x] **TRAIN-03**: Training streams episode rewards via WebSocket
- [x] **TRAIN-04**: Trained weights saved to `trained_weights/` directory
- [x] **TRAIN-05**: Mock fallback works when SUMO unavailable

### Simulation Pipeline (SIM)

- [x] **SIM-01**: RL-controlled simulation loads policy and applies actions via TraCI
- [x] **SIM-02**: Simulation streams frames with vehicle positions and TLS states
- [x] **SIM-03**: Fixed-timing baseline uses SUMO default phases

### Evaluation Framework (EVAL)

- [ ] **EVAL-01**: Evaluation runner orchestrates campaigns across trainers x topologies
- [ ] **EVAL-02**: Metrics: avg waiting time, avg travel time, throughput from SUMO tripinfo
- [ ] **EVAL-03**: Metrics: mean episode reward, communication cost by type
- [ ] **EVAL-04**: Cross-topology transfer matrix (train on X, test on Y)
- [ ] **EVAL-05**: Monte Carlo runs with seed control (target: 10 per config)
- [ ] **EVAL-06**: Persistent results in `BackEnd/results/` as JSON
- [ ] **EVAL-07**: `POST /api/evaluate` starts evaluation job
- [ ] **EVAL-08**: `WS /ws/evaluate/{job_id}` streams per-trial results
- [ ] **EVAL-09**: `GET /api/evaluation/{job_id}` returns aggregated results
- [ ] **EVAL-10**: Fixed-time and max-pressure baselines included in evaluation
- [ ] **EVAL-11**: Transfer gap metric computed across topologies

### Core Extensions (EXT)

- [x] **EXT-01**: FedProx aggregation with proximal term `mu/2 * ||w - w_global||^2`
- [x] **EXT-02**: Cooperative reward shaping with configurable alpha
- [x] **EXT-03**: Time-of-day demand curriculum (AM rush, midday, PM rush)
- [x] **EXT-04**: Sine/cosine time encoding added to observations

### Advanced Extensions (ADV)

- [ ] **ADV-01**: GNN policy replacing MLP for spatial-aware decisions
- [ ] **ADV-02**: Differential privacy with Gaussian noise and (epsilon, delta) tracking
- [ ] **ADV-03**: Emergency vehicle preemption with green phase override

### Frontend Enhancements (UI)

- [ ] **UI-01**: Communication page wired to real evaluation comm cost data
- [ ] **UI-02**: Training page shows per-client reward breakdown
- [ ] **UI-03**: Training history: list past runs, reload/compare
- [ ] **UI-04**: New Evaluation page with controls, results table, grouped bar charts
- [ ] **UI-05**: Transfer matrix heatmap on Evaluation page
- [ ] **UI-06**: Enhanced Compare page with trainer dropdown cycling
- [ ] **UI-07**: Index page fetches real stats from evaluation API
- [ ] **UI-08**: Emergency vehicles rendered in distinct color (blue)

### Experiment Campaigns (CAMP)

- [x] **CAMP-01**: Reproduce baseline paper results: FedRL vs MARL vs SARL vs fixed-time on grid-3x3, grid-5x5
- [ ] **CAMP-02**: FedProx ablation: mu in {0.0, 0.01, 0.1} with fresh training and MC evaluation
- [ ] **CAMP-03**: Cooperative reward ablation: alpha in {1.0, 0.5, 0.1} with fresh training and MC evaluation
- [ ] **CAMP-04**: Time-of-day ablation: fixed demand vs curriculum with time encoding
- [x] **CAMP-05**: Statistical rigor: 10 MC seeds per config, 95% CI, Wilcoxon significance test
- [ ] **CAMP-06**: Publishable output: comparison tables (LaTeX), bar charts with error bars
- [x] **CAMP-07**: Results persistence: all configs and seeds logged alongside results as JSON

### Pre-Experiment Hardening (PRE)

- [ ] **PRE-01**: Multiple training seeds — campaign runner accepts `training_seeds` list, runs each config N times with different RLlib seeds
- [ ] **PRE-02**: Multi-demand configs — campaign configs generate low (150), medium (360), high (600) VPLPH variants
- [ ] **PRE-03**: Fix throughput metric — replace hardcoded 1.0 with real vehicle count from tripinfo
- [ ] **PRE-04**: Fix communication cost accounting — measure actual bytes per aggregation round, not best-effort env logging
- [ ] **PRE-05**: Bootstrap confidence intervals — replace normal approximation CI with bootstrap (n=10 is too small for z-intervals)
- [ ] **PRE-06**: Effect size reporting — add Cohen's d or rank-biserial r alongside Wilcoxon p-values

### New Training Strategies (STRAT)

- [ ] **STRAT-01**: Gossip RL — per-agent policies with peer-to-peer neighbor-only weight averaging (no central server)
- [ ] **STRAT-02**: Mean Field RL — per-agent policies with mean neighbor action appended to observation vector
- [ ] **STRAT-03**: CTDE — centralized critic (global state) with decentralized per-agent actors

### Post-Experiment Analysis (POST)

- [ ] **POST-01**: Communication-performance Pareto curves — plot comm cost vs performance for each strategy
- [ ] **POST-02**: Per-intersection fairness analysis — Gini coefficient or max/min reward ratio across intersections
- [ ] **POST-03**: Demand heterogeneity analysis — how strategy ranking changes across demand levels
- [ ] **POST-04**: Training seed variance — separate train-time variance from eval-time variance in reporting
- [ ] **POST-05**: Mechanistic explanation — why FedRL/MARL/SARL differ (aggregation-as-regularization, specialization tradeoff)
- [ ] **POST-06**: Wall-clock and sample efficiency comparison — time-to-threshold and episodes-to-threshold
- [ ] **POST-07**: Updated experiment report with all new analyses for paper draft

## v2 Requirements

### Advanced UI

- **UI2-01**: Policy inspector page with observation features and network weights
- **UI2-02**: Live communication cost meter during FedRL training
- **UI2-03**: Four-quadrant Compare layout for presentations

## Out of Scope

| Feature | Reason |
|---------|--------|
| CityFlow simulator | Committed to SUMO with existing TraCI integration |
| Docker/containerization | Local development workflow sufficient |
| Multi-user authentication | Single researcher use case |
| Real-time collaboration | Not needed for research workflow |
| Mobile app | Desktop-only research tool |

## Traceability

| Requirement | Phase | Status |
|-------------|-------|--------|
| ENV-01 | Phase 0 | Complete |
| ENV-02 | Phase 0 | Complete |
| ENV-03 | Phase 1 | Complete |
| TRAIN-01 | Phase 1-2 | Complete |
| TRAIN-02 | Phase 2 | Complete |
| TRAIN-03 | Phase 2 | Complete |
| TRAIN-04 | Phase 2 | Complete |
| TRAIN-05 | Phase 2 | Complete |
| SIM-01 | Phase 2 | Complete |
| SIM-02 | Phase 2 | Complete |
| SIM-03 | Phase 2 | Complete |
| EVAL-01 | Phase 3 | Pending |
| EVAL-02 | Phase 3 | Pending |
| EVAL-03 | Phase 3 | Pending |
| EVAL-04 | Phase 3 | Pending |
| EVAL-05 | Phase 3 | Pending |
| EVAL-06 | Phase 3 | Pending |
| EVAL-07 | Phase 3 | Pending |
| EVAL-08 | Phase 3 | Pending |
| EVAL-09 | Phase 3 | Pending |
| EVAL-10 | Phase 3 | Pending |
| EVAL-11 | Phase 3 | Pending |
| EXT-01 | Phase 4 | Complete |
| EXT-02 | Phase 4 | Complete |
| EXT-03 | Phase 4 | Complete |
| EXT-04 | Phase 4 | Complete |
| ADV-01 | Phase 5 | Pending |
| ADV-02 | Phase 5 | Pending |
| ADV-03 | Phase 5 | Pending |
| UI-01 | Phase 6 | Pending |
| UI-02 | Phase 6 | Pending |
| UI-03 | Phase 6 | Pending |
| UI-04 | Phase 6 | Pending |
| UI-05 | Phase 6 | Pending |
| UI-06 | Phase 6 | Pending |
| UI-07 | Phase 6 | Pending |
| UI-08 | Phase 6 | Pending |
| CAMP-01 | Phase 7 | Complete |
| CAMP-02 | Phase 7 | Pending |
| CAMP-03 | Phase 7 | Pending |
| CAMP-04 | Phase 7 | Pending |
| CAMP-05 | Phase 7 | Complete |
| CAMP-06 | Phase 7 | Pending |
| CAMP-07 | Phase 7 | Complete |
| PRE-01 | Phase 9 | Pending |
| PRE-02 | Phase 9 | Pending |
| PRE-03 | Phase 9 | Pending |
| PRE-04 | Phase 9 | Pending |
| PRE-05 | Phase 9 | Pending |
| PRE-06 | Phase 9 | Pending |
| POST-01 | Phase 10 | Pending |
| POST-02 | Phase 10 | Pending |
| POST-03 | Phase 10 | Pending |
| POST-04 | Phase 10 | Pending |
| POST-05 | Phase 10 | Pending |
| POST-06 | Phase 10 | Pending |
| POST-07 | Phase 10 | Pending |

**Coverage:**
- v1 requirements: 56 total
- Mapped to phases: 56
- Unmapped: 0

---
*Requirements defined: 2026-03-22*
*Last updated: 2026-04-06 after pre/post experiment phases added*
