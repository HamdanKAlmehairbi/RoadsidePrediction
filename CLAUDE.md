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

## GSD Planning
- Phase 9: Pre-experiment hardening (methodology fixes + new trainers) — IN PROGRESS
- Phase 10: HPC experiments (Tier 1: 270 runs, Tier 2: TBD)
- Phase 11: Post-experiment analysis (Pareto curves, fairness, mechanistic explanation)

## Ports
- Backend: `http://localhost:8000`
- Frontend: `http://localhost:5173`
