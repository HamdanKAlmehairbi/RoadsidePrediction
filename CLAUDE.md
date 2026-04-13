# RL Training Strategy Benchmarking — Agent Instructions

## Project Context
This is a **benchmarking framework** for evaluating RL training strategies for multi-intersection traffic signal control. 10 strategies are compared across a spectrum from fully independent to fully shared — same environment, same observations, same reward, same algorithm — isolating the training strategy as the only variable.

## Read These Files First
Before doing any work, read:
- `CLAUDE.md` — this file (architecture, strategies, key files)
- `.planning/ROADMAP.md` — phase plan and progress
- `.planning/STATE.md` — current position
- `file-structure.md` — live project tree; **update as you create files**
- `tasks/todo.md` — current status
- `tasks/paper-targets.md` — paper submission task list (ITSC / T-ITS / NeurIPS D&B tiers)
- `CODEX-AUDIT.md` — experimental design audit findings

## Hard Rules
- `SUMO-FedRL-main/` — **never modify**. Already copied into BackEnd.
- `LovableOutput/` — **never modify**. Already copied into FrontEnd.
- `BackEnd/` is owned exclusively by the backend agent.
- `FrontEnd/` is owned exclusively by the frontend agent.

## Architecture
```
LovableOutput/ (read-only)  →  FrontEnd/  ←WebSocket/REST→  BackEnd/  ←TraCI→  SUMO
SUMO-FedRL-main/ (read-only) →  BackEnd/
```

## Benchmarking Framework — Controlled Comparison

The framework controls 8 layers for fair comparison:

1. **Same simulator** — SUMO, same physics, same binary
2. **Same network** — identical .net.xml loaded for all strategies
3. **Same demand** — identical VPLPH, same randomTrips parameters, same seeds
4. **Same observations** — 14-feature intersection-agnostic vector (+ optional augmentation for MeanField/CTDE)
5. **Same reward** — r = -(o + h)^2, shared code path
6. **Same algorithm** — PPO with identical hyperparameters (lr=0.001, gamma=0.95, RLlib defaults)
7. **Same training budget** — identical episode count per config, variable training seeds
8. **Same evaluation** — Monte Carlo with bootstrap CI, Wilcoxon signed-rank with effect size

## Training Strategies (10 total)

```
Independent ←————————————————————————————————————————→ Shared

MARL → MeanField → CTDE → Gossip → HierFed → FedDistill → FedRL → SARL
 (0)    (obs)    (critic) (mesh)   (tree)    (logits)    (star)  (full)
```

| # | Strategy | Trainer Type | File | What makes it unique |
|---|----------|-------------|------|---------------------|
| 1 | SARL | `SARL` | `seal/trainer/single_agent.py` | One shared policy for all intersections |
| 2 | MARL | `MARL` | `seal/trainer/multi_agent.py` | Independent per-intersection policies, no sharing |
| 3 | FedRL | `FedRL` | `seal/trainer/fed_agent.py` | Central server FedAvg (star topology) |
| 4 | Gossip | `Gossip` | `seal/trainer/gossip_agent.py` | Peer-to-peer neighbor weight averaging (mesh) |
| 5 | HierFed | `HierFed` | `seal/trainer/hierfed_agent.py` | Two-tier cluster-then-global averaging (tree) |
| 6 | FedDistill | `FedDistill` | `seal/trainer/feddistill_agent.py` | Share action logits not weights (KL distillation) |
| 7 | MeanField | `MeanField` | `seal/trainer/mean_field_agent.py` | Obs augmented with mean neighbor action |
| 8 | CTDE | `CTDE` | `seal/trainer/ctde_agent.py` | Centralized critic (global state), decentralized actors |
| 9 | fixed-time | `fixed-time` | (eval only) | Always stay in current phase (non-RL floor) |
| 10 | max-pressure | `max-pressure` | (eval only) | Switch to highest queue pressure phase |

### Supporting Files
- `seal/trainer/fedprox_policy.py` — FedProx custom PPO policy (proximal loss term)
- `seal/trainer/feddistill_policy.py` — FedDistill custom PPO policy (KL loss toward consensus)
- `seal/sumo/mean_field_env.py` — Mean Field env wrapper (+1 obs dim for mean neighbor action)
- `seal/sumo/ctde_env.py` — CTDE env wrapper (+global state obs during training, zero-padded at eval)
- `seal/trainer/weight_aggr.py` — Aggregation functions: naive, reward-weighted (shift-normalize), traffic-weighted

### Federated Strategies (4 of 10)
| Strategy | Topology | Shared Payload |
|----------|----------|---------------|
| Gossip | Mesh (neighbors only) | Full weights |
| HierFed | Tree (cluster → global) | Full weights |
| FedDistill | Star (central) | Action logits only |
| FedRL | Star (central) | Full weights |

## Experiment Design

### Tier 1: Core Strategy Comparison (HPC batch 1)
10 strategies × 3 topologies (grid-3x3, grid-5x5, cologne-8) × 3 demands (150, 360, 600 VPLPH) × 3 training seeds = **270 runs**

```bash
cd BackEnd && python scripts/run_extension_ablation.py \
    --ablation strategy \
    --topologies grid-3x3 grid-5x5 cologne-8 \
    --demand-levels 150 360 600 \
    --training-seeds 42 123 456 \
    --n-episodes 50 --n-eval-runs 10
```

### Tier 2: Targeted Ablations (after Tier 1 results)
Designed based on Tier 1 findings. Candidates: aggregation rule, FedProx mu, cooperative alpha, time-of-day, gossip radius, CTDE critic scope.

## Key Infrastructure Files
- `api/training_runner.py` — Trainer factory (`create_trainer()`), topology map, PPO config
- `api/evaluation/runner.py` — `run_trial()` handles all 10 trainer types including multi-policy eval
- `api/evaluation/monte_carlo.py` — MC evaluation with bootstrap CI
- `api/evaluation/metrics.py` — Throughput from tripinfo, comm cost tracking
- `api/evaluation/campaign_config.py` — `ExtensionConfig` dataclass with training_seed, demand, alpha
- `scripts/run_extension_ablation.py` — All 6 ablation builders + CLI
- `scripts/run_campaign.py` — `train_and_evaluate()`, `save_campaign_results()` (append mode)
- `scripts/generate_tables.py` — Wilcoxon + effect size + Bonferroni correction
- `scripts/generate_report.py` — Full experiment report generation

## Statistical Methods
- Bootstrap 95% CI (10,000 resamples, deterministic seed)
- Wilcoxon signed-rank test with rank-biserial effect size
- Bonferroni correction via `apply_bonferroni()` for multiple comparisons
- 3 training seeds per config to separate training variance from eval variance

## Known Patterns (avoid regressions)
- FedProx/FedDistill `__init__` must set custom attributes BEFORE `super().__init__()` (TorchPolicyV2 calls loss() during setup)
- `save_test_policy()` must save `__multi_policy__` format for MARL/Gossip/HierFed/FedDistill/CTDE/MeanField to preserve per-agent specialization
- CTDE eval zero-pads global state portion; `__ctde__` flag in pickle triggers this in runner.py
- MeanField eval uses `MeanFieldSumoEnv` to produce augmented observations
- `save_campaign_results()` appends to existing results.json (not overwrites) for multi-sweep runs
- `training_data` initialized in `BaseTrainer.__init__()` so subclasses can write to it before `on_setup()`
- FedRL `episode_data` reset after each aggregation to avoid cumulative weighting bias

## Experiment Findings (grid-3x3 complete, cologne-8 needs retest)

### grid-3x3 (seed 42, 30 episodes, horizon 450) — COMPLETE
- **All RL strategies crush baselines** by ~80-87% on avg wait time
- Best by demand: FedRL (d150, 9.18s) → Gossip (d360, 11.29s) → SARL (d600, 14.90s)
- **HierFed is most consistent** — top-3 at every demand level (mean 12.04s)
- **CTDE is consistently worst** RL strategy (14.69/15.29/20.58s) — centralized critic hurts on homogeneous grids
- fixed-time baseline: 71.96/75.89/91.82s; max-pressure: 147.76/160.54/143.60s
- `sarl_d150` failed due to temp-dir race condition (now fixed in `abstract_env.py`)
- Results: `results/campaigns/strategy-comparison/grid-3x3/results.json`

### cologne-8 (seed 42, 30 episodes, horizon 450) — NEEDS RETEST
- **RL appears to lose to fixed-time** (RL best: HierFed 54-62s vs fixed-time 39-50s)
- **BUT this is a survivorship bias / horizon truncation problem:**
  - fixed-time completes only ~168 trips; RL completes ~275-303 trips (1.6-1.8x more)
  - fixed-time's 78s arterial green phase serves main road fast but **starves side streets**
  - Unfinished vehicles (40-50% of demand under fixed-time) aren't counted in avg_wait
  - `avg_waiting_time = total_wait / completed_trips` — fewer completions = artificially lower average
  - When penalized for unserved vehicles, SARL beats fixed-time at d360, HierFed ties at d150
- **Convergence curves show RL hasn't converged** at 30 episodes on cologne-8 (still climbing)
- **Drive time inflation**: RL adds 22-38s of pure driving time over fixed-time, indicating spillback from RL phase decisions flooding short downstream links (12m-601m lane length range)
- cologne-8 has **heterogeneous TLS** (4-8 phases, one with 78s dominant green) vs grid-3x3's uniform [42,3,42,3]

### cologne-8 Retest Plan
Retest with `--n-episodes 100 --horizon 900` to fix both issues:
1. **horizon 900** (was 450): lets all vehicles complete so wait metrics are fair across strategies
2. **100 episodes** (was 30): gives RL enough training to converge on the harder network
```bash
cd BackEnd && python scripts/run_extension_ablation.py \
    --ablation strategy --topologies cologne-8 \
    --demand-levels 150 360 600 --training-seeds 42 \
    --n-episodes 100 --n-eval-runs 5 --horizon 900
```
**Clear old results first** to avoid mixing 30ep/100ep data:
```bash
rm BackEnd/results/campaigns/strategy-comparison/cologne-8/results.json
rm BackEnd/results/campaigns/strategy-comparison/cologne-8/config.json
```

### Methodological Findings (important for paper)
1. **`avg_waiting_time` is misleading** when trip completion rates differ — strategies that starve side streets show artificially low averages. Report trip count alongside wait, or use total person-delay.
2. **Synthetic grid results don't transfer** to real networks — RL dominates grid-3x3 but struggles on cologne-8. Papers testing only on grids overstate RL effectiveness.
3. **HierFed's cluster-then-global averaging** is the most robust approach across both topologies — it groups similar intersections before averaging, naturally handling heterogeneity.
4. **Training budget interacts with network complexity** — 30 episodes suffices for grid-3x3 (converges by episode 15) but not cologne-8 (8 heterogeneous intersections, complex spillback dynamics).

## Known Patterns (avoid regressions)
- `demand_dir` in `base.py` is `d{vplph}_s{training_seed}` — weights are scoped by demand AND seed
- `abstract_env.py` cleanup is in `__del__` not `close()` — moving it back to `close()` will re-trigger the SARL race condition
- `run_extension_ablation.py` saves incrementally (one config at a time) and supports crash-resume via `_load_completed_names()`
- Config names always include `_d{demand}_s{seed}` suffix for unique resume keys

## GSD Planning
- Phase 9: Pre-experiment hardening (methodology fixes + new trainers) — IN PROGRESS
- Phase 10: HPC experiments (Tier 1: 270 runs, Tier 2: TBD)
- Phase 11: Post-experiment analysis (Pareto curves, fairness, mechanistic explanation)

## Ports
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
