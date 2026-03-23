# SEAL Dashboard — FedRL Traffic Signal Control

## What This Is

A full-stack platform for federated reinforcement learning experiments on multi-intersection traffic signal control. React frontend visualizes training/simulation, FastAPI backend runs SUMO-based RL experiments using the SEAL framework (SMARTCOMP 2022 paper). Target audience: researcher producing publishable FedRL results.

## Core Value

Real FedRL experiments working end-to-end with publishable results — training, evaluation, and cross-topology transfer on SUMO networks.

## Requirements

### Validated

- ✓ **ENV-01**: SUMO + TraCI + Ray 2.x + gymnasium environment works — Phase 0
- ✓ **ENV-02**: SEAL engine modernized to Ray 2.x with old API stack mode — Phase 1
- ✓ **API-01**: Real FedRL/MARL/SARL training streams rewards via WebSocket — Phase 2
- ✓ **API-02**: RL-controlled simulation with policy inference and TraCI — Phase 2
- ✓ **API-03**: Weight management scans both example_weights and trained_weights — Phase 2
- ✓ **API-04**: SUMO_HOME auto-detection for portable deployment — Phase 2
- ✓ **EVAL-01**: Evaluation runner executes trials across trainers and topologies — Phase 3
- ✓ **EVAL-02**: Metrics computation (wait time, travel time, throughput, comm cost) — Phase 3
- ✓ **EVAL-03**: Cross-topology transfer testing matrix — Phase 3
- ✓ **EVAL-04**: Monte Carlo run management with seed control — Phase 3
- ✓ **EVAL-05**: Persistent results storage (JSON files) — Phase 3
- ✓ **EVAL-06**: API endpoints for evaluation jobs — Phase 3
- ✓ **EXT-01**: FedProx aggregation with proximal term — Phase 4
- ✓ **EXT-02**: Cooperative reward shaping with alpha parameter — Phase 4
- ✓ **EXT-03**: Time-of-day demand curriculum — Phase 4
- ✓ **EXT-04**: Time encoding (sin/cos) in observations — Phase 4

### Active
- [ ] **ADV-01**: GNN policy as alternative to MLP
- [ ] **ADV-02**: Differential privacy on FedAvg
- [ ] **ADV-03**: Emergency vehicle preemption
- [ ] **UI-01**: Wire Communication page to real evaluation data
- [ ] **UI-02**: Training page with per-client reward breakdown
- [ ] **UI-03**: New Evaluation page with results table and charts
- [ ] **UI-04**: Enhanced Compare page with trainer cycling
- [ ] **UI-05**: Index page with real stats from evaluations

### Out of Scope

- CityFlow simulator — staying with SUMO (existing code, TraCI integration works)
- Docker deployment — local development only for now
- Multi-user auth — single researcher workflow
- Real-time collaborative editing — not needed

## Context

- Project replicates SEAL (SMARTCOMP 2022) federated RL paper for multi-intersection traffic signal control
- Core stack: SUMO simulator, Ray RLLib 2.x PPO (old API stack), PyTorch 2.x, Python 3.11
- 6-layer architecture: simulation → environment → observation (14-16 features with time encoding) → binary actions → reward → PPO policies
- FedAvg aggregation with isolated client simulations sharing only model weights
- Evaluation targets: FedLight benchmark datasets (Syn_1/2/3, Real_1) across grid topologies
- Key innovation: intersection-agnostic representation enabling cross-topology transfer

## Constraints

- **Simulator**: SUMO with TraCI — existing code investment, not switching
- **Ray API**: Using old API stack mode (`enable_rl_module_and_learner=False`) for backward compatibility
- **Action Space**: Discrete(2) — binary phase switching per intersection
- **Python**: 3.10-3.11 (Ray 2.x requirement)
- **Platform**: Windows 11, SUMO installed locally

## Key Decisions

| Decision | Rationale | Outcome |
|----------|-----------|---------|
| Stay with SUMO over CityFlow | Existing TraCI integration, route configs already built | ✓ Good |
| Ray 2.x old API stack mode | Minimizes migration risk while getting modern Ray | ✓ Good |
| Discrete(2) action space | Matches SEAL paper binary phase switching | ✓ Good |
| SUMO_HOME auto-detection | Portability without manual env setup | ✓ Good |
| Singleton Ray init at server startup | Avoids per-request init overhead | ✓ Good |
| Frame drop on WebSocket queue | Prevents backpressure blocking training loop | ✓ Good |

---
*Last updated: 2026-03-23 after Phase 4 completion*
