# SEAL: Training Strategy Benchmarking for Multi-Intersection Traffic Signal Control

A standardized evaluation framework comparing **10 RL training strategies** for multi-intersection traffic signal control under identical conditions using the SUMO traffic simulator. The only variable is the training strategy — same environment, observations, reward, algorithm (PPO), and evaluation protocol.

## The Problem

Traffic congestion costs the US $87B annually. RL can train adaptive traffic lights, but the literature compares strategies using different setups, making cross-paper comparison impossible. We built a framework that isolates the training strategy as the **sole variable** across a spectrum from fully independent to fully shared learning.

## 10 Training Strategies

```
Independent <---------------------------------------------------------> Shared

MARL -> MeanField -> CTDE -> Gossip -> HierFed -> FedDistill -> FedRL -> SARL
 (0)     (obs)    (critic)   (mesh)    (tree)     (logits)     (star)  (full)
```

| # | Strategy | What It Does | Communication |
|---|----------|-------------|:---:|
| 1 | **MARL** | Independent per-intersection policies, no sharing | Zero |
| 2 | **MeanField** | Independent policies + neighbor mean action in observation | Zero |
| 3 | **CTDE** | Centralized critic during training, decentralized execution | Zero (runtime) |
| 4 | **Gossip** | Peer-to-peer weight averaging with road-topology neighbors | Moderate |
| 5 | **HierFed** | Two-tier: spatial cluster averaging then global averaging | Moderate |
| 6 | **FedDistill** | Share action logits (not weights) via KL distillation | Low (~100x less) |
| 7 | **FedRL** | Central server FedAvg (star topology) | High |
| 8 | **SARL** | One shared policy for all intersections | N/A (single policy) |
| 9 | **fixed-time** | Fixed phase cycling (non-RL baseline) | Zero |
| 10 | **max-pressure** | Greedy queue-pressure heuristic (non-RL baseline) | Zero |

## Results

Baseline campaign: 10 trainers x 3 topologies x 10 MC seeds = 300 evaluation trials.

### grid-3x3 (9 intersections)

| Rank | Strategy | Avg Waiting Time (s) |
|:----:|----------|:--------------------:|
| 1 | **HierFed** | **11.33** |
| 2 | Gossip | 11.42 |
| 3 | FedDistill | 11.81 |
| ... | ... | ... |
| 9 | fixed-time | 73.64 |
| 10 | max-pressure | 170.68 |

### grid-5x5 (25 intersections)

| Rank | Strategy | Avg Waiting Time (s) |
|:----:|----------|:--------------------:|
| 1 | **Gossip** | **17.60** |
| 2 | SARL | 19.92 |
| 3 | HierFed | 21.02 |
| ... | ... | ... |
| 9 | fixed-time | 70.62 |
| 10 | max-pressure | 152.34 |

### cologne-8 (real-world network)

| Rank | Strategy | Avg Waiting Time (s) |
|:----:|----------|:--------------------:|
| 1 | **fixed-time** | **44.35** |
| 2 | max-pressure | 51.11 |
| 3 | SARL | 54.71 |

RL strategies underperform baselines on cologne-8 at 50 training episodes. Extended training (200 episodes) is being tested on HPC.

### Key Findings

- **Topology-aware coordination wins**: Gossip (avg rank 2.7 across topologies) and HierFed (3.3) consistently outperform both fully independent and fully centralized approaches on synthetic grids.
- **All RL crushes baselines on grids**: 5-15x lower waiting time than fixed-time/max-pressure.
- **Real-world networks need more training**: 50 episodes insufficient for cologne-8's irregular geometry.
- **100% throughput everywhere**: Strategies differ in quality (wait time), not capacity.

See `Report.md` for the full analysis with cross-topology rankings and ablation plans.

## Architecture

```
FrontEnd/ (React + Vite)  <-- REST + WebSocket -->  BackEnd/ (FastAPI + Ray RLlib)  <-- TraCI -->  SUMO
     localhost:5173                                      localhost:8000
```

## 4 Topologies

| Network | Intersections | Type |
|---------|:---:|------|
| grid-3x3 | 9 | Synthetic grid |
| grid-5x5 | 25 | Synthetic grid (scalability test) |
| grid-7x7 | 49 | Synthetic grid (HPC only) |
| cologne-8 | 8 | Real-world (RESCO benchmark, Cologne, Germany) |

## Quick Start

### Prerequisites

- Python 3.12 (Ray does not support 3.13)
- SUMO >= 1.20.0
- Node.js 18+ (frontend only)

### Backend

```bash
cd BackEnd
pip install -r requirements.txt

# Start API server
python -m uvicorn api.main:app --reload --port 8000
```

### Run Experiments

```bash
cd BackEnd

# Train specific trainers on specific topologies (resumable)
python scripts/train_missing_trainers.py \
    --trainers Gossip HierFed FedRL \
    --topologies grid-3x3 grid-5x5 \
    --num-workers 8

# Run evaluation campaign (incremental saves, resumable)
python scripts/run_campaign.py \
    --campaign baseline \
    --topologies grid-3x3 grid-5x5 cologne-8 \
    --n-eval-runs 10 \
    --output-name baseline_full

# Run ablations (demand sweep, fed_step, alpha, fedprox)
python hpc/run_ablation.py --ablation demand
python hpc/run_ablation.py --ablation all
```

### HPC (4 GPUs, ~12 hours)

```bash
cd BackEnd
bash hpc/run_all.sh
```

Runs 7 phases: grid-7x7 training, cologne-8 extended training (200 ep), demand sweep, fed_step/alpha/fedprox ablations. See `hpc/README.md`.

### Frontend

```bash
cd FrontEnd
npm install
npm run dev    # http://localhost:5173
```

## Project Structure

```
BackEnd/
├── seal/                        # SEAL RL framework
│   ├── trainer/
│   │   ├── base.py              # Abstract base trainer
│   │   ├── single_agent.py      # SARL
│   │   ├── multi_agent.py       # MARL
│   │   ├── fed_agent.py         # FedRL (FedAvg, star topology)
│   │   ├── gossip_agent.py      # Gossip (peer-to-peer mesh)
│   │   ├── hierfed_agent.py     # HierFed (two-tier tree)
│   │   ├── feddistill_agent.py  # FedDistill (KL distillation)
│   │   ├── mean_field_agent.py  # MeanField (augmented obs)
│   │   ├── ctde_agent.py        # CTDE (centralized critic)
│   │   ├── fedprox_policy.py    # FedProx custom PPO policy
│   │   ├── feddistill_policy.py # FedDistill custom PPO policy
│   │   └── weight_aggr.py       # Aggregation functions
│   └── sumo/
│       ├── env.py               # Multi-agent SUMO environment
│       ├── ctde_env.py          # CTDE env (global state augmentation)
│       ├── mean_field_env.py    # MeanField env (neighbor actions)
│       └── kernel/              # TraCI interface layer
├── api/
│   ├── training_runner.py       # Trainer factory + training loop
│   └── evaluation/              # Monte Carlo eval, metrics, campaign config
├── configs/SMARTCOMP/           # SUMO .net.xml network files
├── scripts/
│   ├── train_missing_trainers.py # Train specific trainers (parallel GPU support)
│   ├── run_campaign.py          # Evaluation campaign (incremental saves)
│   └── generate_tables.py       # Statistical analysis + LaTeX tables
├── hpc/
│   ├── run_all.sh               # Master HPC script (7 phases)
│   └── run_ablation.py          # Ablation campaign runner
└── results/campaigns/           # Experiment results (JSON)

FrontEnd/
└── src/
    ├── pages/                   # 5 dashboard pages
    ├── components/SimCanvas.tsx # Canvas2D traffic renderer
    └── hooks/                   # WebSocket streaming hooks
```

## Controlled Comparison (8 Layers)

Every experiment controls these factors so the training strategy is the only variable:

1. **Same simulator** — SUMO, same physics
2. **Same network** — identical .net.xml
3. **Same demand** — identical VPLPH, same seeds
4. **Same observations** — 14-feature vector per intersection
5. **Same reward** — r = -(occupancy + halted_occupancy)^2
6. **Same algorithm** — PPO (lr=0.001, gamma=0.95)
7. **Same training budget** — 50 episodes per config
8. **Same evaluation** — 10 Monte Carlo seeds with bootstrap 95% CI

## References

- Ye et al. (2021) — FedLight: Federated RL for Traffic Signal Control. DAC.
- Hudson et al. (2022) — SEAL: Smart Edge-Enabled Traffic Light Control. SMARTCOMP.
- Bao et al. (2023) — Scalable FL for Traffic Signal Control. Scientific Reports.
- Li et al. (2020) — Federated Optimization in Heterogeneous Networks (FedProx). MLSys.
- Ault & Sharon (2021) — RESCO: RL Benchmarks for Traffic Signal Control. NeurIPS.
- McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks (FedAvg). AISTATS.
