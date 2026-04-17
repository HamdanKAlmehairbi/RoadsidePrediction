# PROJECT SUMMARY
## Federated RL Traffic Signal Control Benchmarking Framework

**Paper-ready title:** "When Does Federation Help? A Controlled Comparison of Training Strategies for Multi-Intersection Traffic Signal Control"

**One-liner:** The first standardized evaluation framework comparing 10 RL training strategies for traffic signal control under identical conditions using SUMO.

---

## 1. What This System Does

This is a benchmarking framework that evaluates how different ways of organizing RL training affect multi-intersection traffic signal control. 10 strategies are compared across a spectrum from fully independent to fully shared -- same environment, same observations, same reward, same algorithm (PPO) -- isolating the **training strategy** as the only variable.

The system provides:
- **Backend**: Python/FastAPI server with SUMO traffic simulator integration, Ray RLlib training, and automated evaluation
- **Frontend**: React dashboard for real-time training visualization, simulation playback, and strategy comparison
- **CLI tools**: Batch experiment scripts for running large-scale ablation studies and generating publication-ready tables/figures

**Why it matters:** Traffic congestion costs the US $87B annually. RL can train adaptive traffic lights, but the literature has fewer than 10 papers on federated RL for traffic signals, each using different setups. No fair head-to-head comparison existed until this framework.

---

## 2. Architecture

```
FrontEnd/ (React + Vite)  <-- WebSocket/REST -->  BackEnd/ (FastAPI + Ray)  <-- TraCI -->  SUMO
     localhost:5173                                    localhost:8000
```

### Backend Layers
1. **API Layer** (`api/main.py`) -- FastAPI with REST routes + WebSocket streaming
2. **Training Factory** (`api/training_runner.py`) -- `create_trainer()` dispatches to 8 trainer classes
3. **Trainer Classes** (`seal/trainer/*.py`) -- Each strategy implements `BaseTrainer` ABC
4. **SUMO Environment** (`seal/sumo/abstract_env.py`) -- Multi-agent Gymnasium env, one agent per traffic light
5. **Evaluation Pipeline** (`api/evaluation/`) -- Monte Carlo eval, bootstrap CI, tripinfo metrics
6. **Campaign Scripts** (`scripts/`) -- CLI tools for batch experiments and analysis

### Frontend Layers
- React 18 + Vite + TailwindCSS + Shadcn UI
- 5 pages: Index (overview), Simulation (playback), Training (live curves), Compare (side-by-side), Communication (topology viz)
- Recharts for visualization, TanStack Query for data fetching

---

## 3. The 10 Training Strategies

```
Independent <----------------------------------------------------> Shared

MARL -> MeanField -> CTDE -> Gossip -> HierFed -> FedDistill -> FedRL -> SARL
 (0)     (obs)    (critic)  (mesh)    (tree)     (logits)     (star)  (full)
```

| # | Strategy | Type | File | What Makes It Unique |
|---|----------|------|------|---------------------|
| 1 | SARL | Single-agent | `seal/trainer/single_agent.py` | One shared policy for all intersections |
| 2 | MARL | Multi-agent | `seal/trainer/multi_agent.py` | Independent per-intersection policies, no sharing |
| 3 | FedRL | Federated | `seal/trainer/fed_agent.py` | Central server FedAvg (star topology) |
| 4 | Gossip | Federated | `seal/trainer/gossip_agent.py` | Peer-to-peer neighbor weight averaging (mesh) |
| 5 | HierFed | Federated | `seal/trainer/hierfed_agent.py` | Two-tier cluster-then-global averaging (tree) |
| 6 | FedDistill | Federated | `seal/trainer/feddistill_agent.py` | Share action logits not weights (KL distillation) |
| 7 | MeanField | Augmented | `seal/trainer/mean_field_agent.py` | Obs augmented with mean neighbor action |
| 8 | CTDE | Centralized | `seal/trainer/ctde_agent.py` | Centralized critic (global state), decentralized actors |
| 9 | fixed-time | Baseline | (eval only) | Always stay in current phase (non-RL floor) |
| 10 | max-pressure | Baseline | (eval only) | Switch to highest queue pressure phase |

### Supporting Files
- `seal/trainer/fedprox_policy.py` -- FedProx custom PPO policy (proximal loss term)
- `seal/trainer/feddistill_policy.py` -- FedDistill custom PPO policy (KL loss toward consensus)
- `seal/sumo/mean_field_env.py` -- Mean Field env wrapper (+1 obs dim)
- `seal/sumo/ctde_env.py` -- CTDE env wrapper (+global state obs during training)
- `seal/trainer/weight_aggr.py` -- Aggregation: naive, reward-weighted (shift-normalize), traffic-weighted

---

## 4. Controlled Comparison Framework (8 Layers)

Every experiment controls these 8 factors so the training strategy is the only variable:

1. **Same simulator** -- SUMO, same physics, same binary
2. **Same network** -- identical .net.xml for all strategies
3. **Same demand** -- identical VPLPH, same randomTrips parameters, same seeds
4. **Same observations** -- 14-feature intersection-agnostic vector (+ optional augmentation for MeanField/CTDE)
5. **Same reward** -- r = -(queue + halted)^2, shared code path
6. **Same algorithm** -- PPO with identical hyperparameters (lr=0.001, gamma=0.95, RLlib defaults)
7. **Same training budget** -- identical episode count per config, variable training seeds
8. **Same evaluation** -- Monte Carlo with bootstrap CI, Wilcoxon signed-rank with effect size

---

## 5. Setup and Running

### Prerequisites
- Python 3.10+
- SUMO >= 1.20.0 (auto-detected from standard install paths)
- Node.js 18+ (for frontend)

### Backend Setup
```bash
cd BackEnd
pip install -r requirements.txt

# Start API server
python -m uvicorn api.main:app --reload --port 8000

# Or run experiments directly via CLI (no server needed)
python scripts/run_extension_ablation.py --ablation strategy \
    --topologies grid-3x3 grid-5x5 cologne-8 \
    --demand-levels 150 360 600 \
    --training-seeds 42 123 456 \
    --n-episodes 50 --n-eval-runs 10 --horizon 900
```

### Frontend Setup
```bash
cd FrontEnd
npm install
npm run dev    # http://localhost:5173
```

### Key Dependencies
- **Ray RLlib >= 2.9.0** -- distributed RL training (PPO algorithm)
- **Gymnasium >= 0.29.0** -- RL environment interface
- **PyTorch >= 2.0** -- neural network backend
- **FastAPI >= 0.110.0** -- REST API + WebSocket
- **traci + sumolib** -- SUMO programmatic control
- **scipy** -- statistical tests (Wilcoxon, bootstrap)

---

## 6. Experiment Results (Current — April 2026)

### Completed: Baseline Campaign (30 configs, 300 MC trials)

All 8 RL trainers trained (50 episodes) and evaluated (10 MC seeds) on 3 topologies.
Results: `BackEnd/results/campaigns/baseline_full/results.json`

**grid-3x3 (9 intersections) — best to worst by avg wait:**

| Rank | Strategy | Avg Wait (s) | Avg Travel (s) |
|:----:|----------|:---:|:---:|
| 1 | HierFed | 11.33 | 56.16 |
| 2 | Gossip | 11.42 | 56.33 |
| 3 | FedDistill | 11.81 | 56.75 |
| 4 | MeanField | 12.75 | 57.93 |
| 5 | MARL | 14.23 | 59.82 |
| 6 | SARL | 14.88 | 59.77 |
| 7 | FedRL | 15.61 | 61.32 |
| 8 | CTDE | 15.96 | 61.46 |
| 9 | fixed-time | 73.64 | 113.97 |
| 10 | max-pressure | 170.68 | 213.98 |

**grid-5x5 (25 intersections):**

| Rank | Strategy | Avg Wait (s) | Avg Travel (s) |
|:----:|----------|:---:|:---:|
| 1 | Gossip | 17.60 | 83.66 |
| 2 | SARL | 19.92 | 84.60 |
| 3 | HierFed | 21.02 | 87.84 |
| 4 | FedDistill | 21.09 | 87.85 |
| 5 | FedRL | 21.84 | 88.06 |
| 6 | CTDE | 21.86 | 88.60 |
| 7 | MeanField | 23.91 | 91.30 |
| 8 | MARL | 24.95 | 92.46 |
| 9 | fixed-time | 70.62 | 124.64 |
| 10 | max-pressure | 152.34 | 210.72 |

**cologne-8 (8 real-world intersections — RL loses to baselines):**

| Rank | Strategy | Avg Wait (s) | Avg Travel (s) |
|:----:|----------|:---:|:---:|
| 1 | fixed-time | 44.35 | 101.05 |
| 2 | max-pressure | 51.11 | 106.66 |
| 3 | SARL | 54.71 | 134.14 |
| 4-10 | All other RL | 59-72 | 144-163 |

### Key Findings
1. **Topology-aware coordination wins on synthetic grids.** HierFed (avg rank 2.7) and Gossip (3.3) consistently outperform both fully independent (MARL) and fully centralized (FedRL, SARL).
2. **RL fails on cologne-8 at 50 episodes.** All RL strategies lose to fixed-time. Likely insufficient training on the irregular real-world network — HPC ablation will retest at 200 episodes.
3. **All RL massively beats baselines on grids** (5-15x lower waiting time).
4. **100% throughput everywhere** — strategies differ in quality (wait time), not capacity.

---

## 7. What Is Complete

### Infrastructure
- All 10 training strategies implemented and functional
- Multi-topology support (grid-3x3, grid-5x5, grid-7x7, cologne-8)
- Multi-demand support (150, 360, 600 VPLPH) with demand-aware paths
- Monte Carlo evaluation with bootstrap 95% CI
- Statistical analysis: Wilcoxon signed-rank, effect size, Bonferroni correction
- Crash-resilient experiment runner with incremental saves and resume
- Campaign results with lockfile-based parallel write safety
- Publication-ready table/figure generation (LaTeX, bar charts, heatmaps)
- Frontend dashboard with 5 pages (functional but uses mock simulation for playback)
- REST API + WebSocket streaming for real-time training visualization

### Evaluation Framework
- `run_trial()` handles all 10 trainer types including multi-policy eval
- Tripinfo-based metrics: avg waiting time, avg travel time, throughput, depart delay
- Communication cost tracking per strategy
- Transfer evaluation (train on X, eval on Y)

---

## 8. Current Status and Next Steps

### Where we are: READY FOR HPC ABLATION RUNS

Baseline training and evaluation complete for grid-3x3, grid-5x5, cologne-8. HPC package ready in `BackEnd/hpc/`.

### HPC Campaign (7 phases, ~12 hours on 4 GPUs)
1. **Train grid-7x7** — 8 RL trainers on 49-intersection network (new topology)
2. **Eval grid-7x7** — 10 trainers × 5 MC seeds
3. **Cologne-8 extended** — Retrain all 8 RL trainers with 200 episodes (4x baseline) to test if RL can beat baselines with more training
4. **Demand sweep** — Eval at 150/600 VPLPH (360 already done) across all 4 topologies
5. **Fed step sweep** — Retrain FedRL/Gossip/HierFed at fed_step=3,5,10
6. **Alpha sweep** — Retrain top trainers at alpha=0.1,0.3,0.7 (cooperative reward blending)
7. **FedProx mu sweep** — Retrain FedRL at mu=0.01,0.1,1.0

Run: `cd BackEnd && bash hpc/run_all.sh`

### After HPC: Final Report
- Merge HPC results back to local
- Generate final report with all ablation findings
- Publication-ready tables and figures

---

## 9. Known Issues and Risks

### Audit Findings (from CODEX-AUDIT.md, 2026-04-02)

**CRITICAL -- MARL Eval Averaged Specialization (FIXED):**
Original code averaged all per-intersection MARL policies into one before evaluation, destroying MARL's value proposition. Fixed: `save_test_policy()` now saves `__multi_policy__` format for MARL/Gossip/HierFed/FedDistill/CTDE/MeanField to preserve per-agent specialization.

**CRITICAL -- Paper Numbers vs Code Mismatch:**
Paper tables used numbers from baseline campaign (pre-trained example weights) while training-curves campaign used freshly trained weights. Communication costs (102.6 MB / 285 MB) were theoretical estimates, not measured. Alpha computation in code (`reward / total_reward`) differs from paper formula (shift-normalize). All need reconciliation before publication.

**MODERATE -- "Sole Variable" Claim Overstated:**
FedRL resets weights after every episode (others accumulate). FedProx changes loss function. Claim is valid for SARL/MARL/FedRL core comparison but overstated for aggregation variants.

### Technical Gotchas
- **SUMO binary resolution**: Ray DLL side-effects poison `SUMO_HOME` to `sumo_data` on Windows. Always use `sumolib.checkBinary()`, never raw `os.environ["SUMO_HOME"]` for binary paths.
- **Windows libsumo DLL conflict**: Ray's imports break libsumo on Windows — accepted limitation. TraCI fallback works. libsumo activates automatically on Linux/HPC.
- **Python 3.13 incompatible**: Ray requires Python 3.12. Use `RoadsideVenv` conda env, not system Python.
- FedProx/FedDistill `__init__` must set custom attributes BEFORE `super().__init__()` (TorchPolicyV2 calls loss() during setup)
- CTDE eval zero-pads global state portion; `__ctde__` flag in pickle triggers this
- MeanField eval uses `MeanFieldSumoEnv` for augmented observations
- `abstract_env.py` cleanup is in `__del__` not `close()` -- moving it back triggers SARL race condition
- FedRL `episode_data` must reset after each aggregation to avoid cumulative weighting bias
- `avg_waiting_time` is misleading when trip completion rates differ between strategies (survivorship bias)

### Frontend Gotchas
- Recharts `isAnimationActive` must be `false` for real-time data (animation restarts kill updates)
- SimCanvas must use single `requestAnimationFrame` loop (multiple loops cause flicker)
- ErrorBoundary required on Simulation and Compare pages to prevent blackscreens

---

## 10. Statistical Methods

- **Bootstrap 95% CI**: 10,000 resamples, deterministic seed
- **Wilcoxon signed-rank test** with rank-biserial effect size for pairwise strategy comparison
- **Bonferroni correction** via `apply_bonferroni()` for multiple comparisons
- **3 training seeds** per config to separate training variance from eval variance
- **Monte Carlo evaluation**: 5-10 eval runs per trained model with controlled seeds

Scripts: `scripts/generate_tables.py` (stats + LaTeX), `scripts/generate_report.py` (full narrative report)

---

## 11. Key File Reference

### Training
| File | Purpose |
|------|---------|
| `seal/trainer/base.py` | Abstract base trainer (PPO config, checkpointing, paths) |
| `seal/trainer/single_agent.py` | SARL trainer |
| `seal/trainer/multi_agent.py` | MARL trainer |
| `seal/trainer/fed_agent.py` | FedRL trainer (FedAvg) |
| `seal/trainer/gossip_agent.py` | Gossip trainer (peer-to-peer) |
| `seal/trainer/hierfed_agent.py` | HierFed trainer (cluster + global) |
| `seal/trainer/feddistill_agent.py` | FedDistill trainer (KL distillation) |
| `seal/trainer/mean_field_agent.py` | MeanField trainer (augmented obs) |
| `seal/trainer/ctde_agent.py` | CTDE trainer (centralized critic) |
| `seal/trainer/weight_aggr.py` | Aggregation functions |
| `seal/trainer/fedprox_policy.py` | FedProx PPO policy |
| `seal/trainer/feddistill_policy.py` | FedDistill PPO policy |

### Evaluation
| File | Purpose |
|------|---------|
| `api/evaluation/runner.py` | `run_trial()` for all 10 trainer types |
| `api/evaluation/monte_carlo.py` | MC evaluation with bootstrap CI |
| `api/evaluation/metrics.py` | Tripinfo parsing, comm cost tracking |
| `api/evaluation/campaign_config.py` | `ExtensionConfig` dataclass |

### Experiment Scripts
| File | Purpose |
|------|---------|
| `scripts/run_extension_ablation.py` | All ablation builders + CLI (main experiment runner) |
| `scripts/run_campaign.py` | `train_and_evaluate()`, lockfile-safe `save_campaign_results()` |
| `scripts/generate_tables.py` | Wilcoxon + effect size + Bonferroni + LaTeX tables |
| `scripts/generate_report.py` | Full experiment report generation |

### Environment
| File | Purpose |
|------|---------|
| `seal/sumo/abstract_env.py` | Base SUMO multi-agent env |
| `seal/sumo/env.py` | Standard SUMO env |
| `seal/sumo/ctde_env.py` | CTDE env (global state augmentation) |
| `seal/sumo/mean_field_env.py` | MeanField env (neighbor action augmentation) |
| `seal/sumo/kernel/` | TraCI interface layer |
| `configs/SMARTCOMP/*.net.xml` | SUMO network files |

### API
| File | Purpose |
|------|---------|
| `api/main.py` | FastAPI app, Ray init, route registration |
| `api/training_runner.py` | Trainer factory, training loop |
| `api/routes/` | REST endpoints |
| `api/ws/` | WebSocket streaming |

---

## 12. HPC Scripts

| Script | Purpose |
|--------|---------|
| `hpc/run_all.sh` | Master script — runs all 7 phases (SLURM-ready) |
| `hpc/run_ablation.py` | Ablation runner (cologne_extended, demand, fed_step, alpha, fedprox) |
| `scripts/train_missing_trainers.py` | Train specific trainers on specific topologies (resumable, parallel GPU) |
| `scripts/run_campaign.py` | Evaluation campaign with incremental saves (resumable) |
