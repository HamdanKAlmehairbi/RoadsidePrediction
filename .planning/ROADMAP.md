# Roadmap: SEAL Dashboard

## Overview

Transform a mock-data dashboard into a real FedRL research platform. Phases 0-2 established the foundation (SUMO, Ray 2.x, API wiring). Remaining phases build the evaluation framework, add research extensions (FedProx, cooperative reward, GNN, DP), and wire the frontend to real data.

## Phases

- [x] **Phase 0: Environment Setup** - Install SUMO, Python deps, verify TraCI
- [x] **Phase 1: Modernize SEAL Engine** - Ray 2.x migration, gymnasium, 5-tuple returns
- [x] **Phase 2: Wire Real Training + Simulation** - Replace stubs with SEAL integration
- [ ] **Phase 3: Evaluation Framework** - Runner, baselines, metrics, transfer, MC runs
- [ ] **Phase 4: Core Extensions** - FedProx, cooperative reward, time-of-day
- [ ] **Phase 5: Advanced Extensions** - GNN policy, differential privacy, emergency preemption
- [ ] **Phase 6: Frontend Enhancements** - Wire all pages to real data, new Evaluation page
- [ ] **Phase 7: Experiment Campaigns** - Reproduce paper, evaluate extensions, publishable results
- [ ] **Phase 8: Experiment Execution & Analysis** - Run all campaigns against live SUMO, generate publishable artifacts
- [ ] **Phase 9: Pre-Experiment Hardening** - Fix methodology gaps: training seeds, multi-demand, real metrics, stronger stats
- [ ] **Phase 10: Post-Experiment Analysis** - Pareto curves, fairness, demand heterogeneity, mechanistic explanations, paper report

## Phase Details

### Phase 0: Environment Setup
**Goal**: Working SUMO + Python environment with all dependencies
**Depends on**: Nothing
**Requirements**: ENV-01, ENV-02
**Success Criteria** (what must be TRUE):
  1. `sumo --version` succeeds
  2. `import traci; import ray; import gymnasium` all succeed
  3. Raw SUMO simulation runs on grid-3x3 network
**Plans**: Complete

Plans:
- [x] 00-01: Install SUMO and configure PATH
- [x] 00-02: Create Python environment with dependencies

### Phase 1: Modernize SEAL Engine
**Goal**: SEAL trainers work with Ray 2.x on modern Python
**Depends on**: Phase 0
**Requirements**: ENV-03, TRAIN-01
**Success Criteria** (what must be TRUE):
  1. `FedPolicyTrainer(...).train(2)` completes on grid-3x3
  2. All SEAL imports succeed without deprecation errors
  3. Episode metrics accessible from Ray 2.x result structure
**Plans**: Complete

Plans:
- [x] 01-01: Migrate environment layer (gym -> gymnasium, 5-tuple)
- [x] 01-02: Migrate trainer base (PPOConfig builder, Ray 2.x APIs)
- [x] 01-03: Migrate FedRL/MARL/SARL trainers and callbacks

### Phase 2: Wire Real Training + Simulation
**Goal**: API endpoints use real SEAL training and RL-controlled simulation
**Depends on**: Phase 1
**Requirements**: TRAIN-02, TRAIN-03, TRAIN-04, TRAIN-05, SIM-01, SIM-02, SIM-03
**Success Criteria** (what must be TRUE):
  1. `POST /api/train` with FedRL streams real reward curves
  2. `POST /api/simulate` with trained weights shows RL-controlled vehicles
  3. Mock fallback still works without SUMO
  4. Weights saved to `trained_weights/` and listed by API
**Plans**: Complete

Plans:
- [x] 02-01: Create training_runner.py wrapper with singleton Ray
- [x] 02-02: Implement _run_real_training() with SEAL integration
- [x] 02-03: Implement RL-controlled simulation with policy inference
- [x] 02-04: Extend weight management and add SUMO_HOME auto-detection

### Phase 3: Evaluation Framework
**Goal**: Automated evaluation campaigns producing publishable results tables
**Depends on**: Phase 2
**Requirements**: EVAL-01, EVAL-02, EVAL-03, EVAL-04, EVAL-05, EVAL-06, EVAL-07, EVAL-08, EVAL-09, EVAL-10, EVAL-11
**Success Criteria** (what must be TRUE):
  1. Full eval: 5 trainers x 3 topologies x 10 MC runs completes
  2. FedRL shows measurable comm reduction vs SARL
  3. Transfer gap measurable across topologies
  4. Results persist as JSON and are queryable via REST
  5. Fixed-time and max-pressure baselines produce valid metrics
**Plans**: 5 plans

Plans:
- [x] 03-01-PLAN.md — Evaluation runner and baselines (run_trial, fixed-time, max-pressure)
- [x] 03-02-PLAN.md — Metrics computation and cross-topology transfer testing
- [x] 03-03-PLAN.md — Monte Carlo orchestration with seed control and aggregation
- [x] 03-04-PLAN.md — API endpoints (POST/GET/WS) and persistent JSON results store
- [x] 03-05-PLAN.md — Integration testing, validation, and package re-exports

### Phase 4: Core Extensions
**Goal**: FedProx, cooperative reward, and time-of-day demand add research depth
**Depends on**: Phase 3 (needs evaluation framework to measure effects)
**Requirements**: EXT-01, EXT-02, EXT-03, EXT-04
**Success Criteria** (what must be TRUE):
  1. FedProx converges faster than FedAvg on heterogeneous clients
  2. Cooperative reward with alpha parameter shows measurable effect
  3. Time-of-day curriculum demonstrates adaptation vs fixed-timing degradation
  4. All extensions selectable via API parameters
**Plans**: 3 plans

Plans:
- [x] 04-01-PLAN.md — FedProx proximal policy and FedPolicyTrainer integration
- [x] 04-02-PLAN.md — Cooperative reward shaping with alpha parameter
- [x] 04-03-PLAN.md — Time-of-day demand curriculum, time encoding, and API wiring for all Phase 4 params

### Phase 5: Advanced Extensions
**Goal**: GNN, differential privacy, and emergency preemption for advanced research
**Depends on**: Phase 3 (needs evaluation framework)
**Requirements**: ADV-01, ADV-02, ADV-03
**Success Criteria** (what must be TRUE):
  1. GNN outperforms MLP on larger grids (5x5, 7x7)
  2. DP shows privacy-utility tradeoff curve
  3. Emergency vehicles get green phase override within detection threshold
**Plans**: TBD

Plans:
- [ ] 05-01: GNN policy as custom Ray model
- [ ] 05-02: Differential privacy on FedAvg
- [ ] 05-03: Emergency vehicle preemption

### Phase 6: Frontend Enhancements
**Goal**: All frontend pages display real data from evaluation and training APIs
**Depends on**: Phase 3 (needs evaluation endpoints)
**Requirements**: UI-01, UI-02, UI-03, UI-04, UI-05, UI-06, UI-07, UI-08
**Success Criteria** (what must be TRUE):
  1. Communication page shows real comm cost breakdown
  2. Training page shows per-client rewards and training history
  3. New Evaluation page renders results table with error bars
  4. Transfer matrix heatmap displays on Evaluation page
  5. Index page stats come from real evaluation data
  6. No console errors on any page
**Plans**: TBD

Plans:
- [ ] 06-01: Wire Communication and Training pages to real data
- [ ] 06-02: Build Evaluation page with results table and charts
- [ ] 06-03: Enhance Compare and Index pages

### Phase 9: Pre-Experiment Hardening
**Goal:** Fix methodology gaps + implement 3 new training strategies (Gossip RL, Mean Field RL, CTDE) to create an 8-way strategy comparison spectrum from fully independent to fully shared
**Depends on:** Phase 8 (uses campaign infrastructure)
**Requirements**: PRE-01, PRE-02, PRE-03, PRE-04, PRE-05, PRE-06, STRAT-01, STRAT-02, STRAT-03
**Success Criteria** (what must be TRUE):
  1. Campaign runner accepts `training_seeds` list and multi-demand configs
  2. Throughput and communication cost metrics are real, not hardcoded
  3. Bootstrap CI and Bonferroni-corrected Wilcoxon with effect size
  4. GossipPolicyTrainer averages weights with topological neighbors only
  5. MeanFieldTrainer augments observation with mean neighbor action
  6. CTDETrainer uses centralized critic with global state, decentralized actors
  7. All 8 strategies run on grid-3x3 without error (smoke test)
**Plans**: 4 plans

Plans:
- [x] 09-01: Training seed support, multi-demand configs, throughput fix, comm cost fix, bootstrap CI, effect size, Bonferroni
- [ ] 09-02: Gossip RL trainer — peer-to-peer neighbor weight averaging
- [ ] 09-03: Mean Field RL trainer — mean neighbor action observation augmentation
- [ ] 09-04: CTDE trainer — centralized critic, decentralized actors

### Phase 10: HPC Experiments
**Goal:** Run two-tier experiment campaign on HPC
**Depends on:** Phase 9
**Experiment Design:**
  - **Tier 1 (core):** 8 strategies × 3 topologies × 3 demands × 3 seeds = 216 runs
  - **Tier 2 (ablations):** Designed after Tier 1 results — targeted ablations on top strategies only
  - Strategies: SARL, MARL, FedRL, Gossip RL, Mean Field, CTDE, fixed-time, max-pressure

Plans:
- [ ] 10-01: Tier 1 experiment execution (216 runs)
- [ ] 10-02: Tier 2 ablation design based on Tier 1 results

### Phase 11: Post-Experiment Analysis
**Goal:** Transform raw HPC results into publishable empirical analysis — Pareto curves, fairness metrics, demand heterogeneity analysis, mechanistic explanations, and paper-ready report
**Depends on:** Phase 10 (HPC results)
**Requirements**: POST-01, POST-02, POST-03, POST-04, POST-05, POST-06, POST-07
**Success Criteria** (what must be TRUE):
  1. Communication-performance Pareto plot shows tradeoff curve across strategies
  2. Per-intersection fairness metric (Gini) computed for each strategy × topology
  3. Strategy ranking table shows how winner changes across demand levels
  4. Training variance separated from eval variance in error reporting
  5. Report includes mechanistic explanation section (not just numbers)
  6. Wall-clock and sample efficiency comparison included
  7. Updated EXPERIMENT_REPORT.md ready for paper draft integration

Plans:
- [ ] 11-01: Pareto curves, fairness analysis, and demand heterogeneity
- [ ] 11-02: Variance decomposition and efficiency analysis
- [ ] 11-03: Mechanistic explanation and final paper-ready report

## Progress

| Phase | Plans Complete | Status | Completed |
|-------|----------------|--------|-----------|
| 0. Environment Setup | 2/2 | Complete | 2026-03-22 |
| 1. Modernize SEAL Engine | 3/3 | Complete | 2026-03-22 |
| 2. Wire Real Training + Simulation | 4/4 | Complete | 2026-03-22 |
| 3. Evaluation Framework | 5/5 | Complete | 2026-03-22 |
| 4. Core Extensions | 3/3 | Complete | 2026-03-23 |
| 5. Advanced Extensions | 0/3 | Not started | - |
| 6. Frontend Enhancements | 0/3 | Not started | - |
| 7. Experiment Campaigns | 3/3 | Complete | 2026-03-23 |
| 9. Pre-Experiment Hardening | 1/4 | In progress | - |
| 10. HPC Experiments | 0/2 | Not started | - |
| 11. Post-Experiment Analysis | 0/3 | Not started | - |

### Phase 7: Experiment Campaigns

**Goal:** Reproduce SEAL paper results, evaluate Phase 4 extensions (FedProx, cooperative reward, time-of-day), and produce publishable comparison tables with statistical rigor
**Requirements**: CAMP-01, CAMP-02, CAMP-03, CAMP-04, CAMP-05, CAMP-06, CAMP-07
**Depends on:** Phase 4 (needs core extensions)
**Success Criteria** (what must be TRUE):
  1. Baseline campaign (FedRL/MARL/SARL/fixed-time/max-pressure on grid-3x3, grid-5x5) produces results JSON
  2. Extension ablation scripts define correct parameter grids (FedProx mu, cooperative alpha, time-of-day)
  3. Wilcoxon signed-rank test compares paired MC results between conditions
  4. LaTeX comparison tables with mean +/- std and booktabs formatting generated
  5. Bar charts with 95% CI error bars generated as PNG
  6. All campaign configs and seeds persisted alongside results
**Plans:** 3 plans

Plans:
- [x] 07-01-PLAN.md — Campaign config, example weights mapping, and baseline runner script
- [x] 07-02-PLAN.md — Extension ablation scripts (FedProx, cooperative reward, time-of-day)
- [x] 07-03-PLAN.md — Statistical analysis, LaTeX tables, and chart generation

### Phase 8: Experiment Execution & Analysis

**Goal:** Run all experiment campaigns (baseline reproduction, FedProx ablation, cooperative reward ablation, time-of-day ablation) against live SUMO, then generate publishable analysis artifacts (LaTeX tables, bar charts, convergence curves, Wilcoxon significance tests, and summary report)
**Requirements**: EXP-01 (baseline reproduction run), EXP-02 (FedProx ablation run), EXP-03 (cooperative ablation run), EXP-04 (time-of-day ablation run), EXP-05 (statistical analysis & figures)
**Depends on:** Phase 7
**Plans:** 2/3 plans executed

Plans:
- [x] 08-01-PLAN.md — Analysis tooling: multi-topology ablation support, convergence curves, combined tables, report generator
- [x] 08-02-PLAN.md — Baseline reproduction campaign (dry-run + full run + user checkpoint)
- [ ] 08-03-PLAN.md — Extension ablation campaigns + full analysis artifact generation + final report

---
*Roadmap created: 2026-03-22*
*Last updated: 2026-03-23 after Phase 8 planning*
