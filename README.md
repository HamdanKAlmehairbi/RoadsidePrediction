# Federated RL Traffic Signal Control — Benchmarking Framework

A standardized evaluation framework for benchmarking reinforcement learning training strategies for multi-intersection traffic signal control. Compares single-agent (SARL), multi-agent (MARL), and federated (FedRL) approaches under identical conditions using the SUMO traffic simulator.

## What This Project Does

Traffic lights in most cities run on fixed timers. Reinforcement learning can train them to adapt to real-time conditions. But there are multiple ways to organize training across a network of intersections, and the existing literature compares them using different setups, making cross-paper comparison impossible.

We built a framework that evaluates three training strategies under controlled conditions:

- **SARL (Single-Agent)** — One shared policy for all intersections. Maximum information sharing, highest communication cost.
- **MARL (Multi-Agent)** — Separate policy per intersection, centralized training coordination. Each agent specializes, but constant communication required.
- **FedRL (Federated)** — Separate policies, local training, periodic weight sharing via an edge server. Agents learn from each other without transmitting raw data.

All three strategies share the same environment, observations, reward function, RL algorithm (PPO), hyperparameters, and evaluation protocol. The only variable is how policies are organized and whether they share weights.

## Key Findings (Midterm)

Under standard demand (360 vehicles/lane/hour):

| Strategy | Grid 3x3 Wait (s) | Grid 5x5 Wait (s) | Communication |
|----------|-------------------:|-------------------:|--------------:|
| Federated | 11.5 | 16.8 | 102.6 MB |
| Centralized | 13.9 | 23.6 | 174.6 MB |
| Decentralized | 10.4 | 17.4 | 57.6 MB |
| Fixed-Time | 76.9 | 70.6 | 0 MB |

- All RL strategies reduce waiting time by 75-85% vs fixed-time control
- Federated RL matches centralized performance with 41% less communication
- Federated's advantage grows with network size (3x3 vs 5x5)

## Architecture

```
FrontEnd/          React + Vite + shadcn/ui + Tailwind + Canvas API
    |  REST + WebSocket
BackEnd/           FastAPI + Ray RLlib + PyTorch (SEAL framework)
    |  TraCI (Python API)
SUMO               Microscopic traffic simulator
```

## Evaluation Framework

The benchmarking framework ensures fair comparison through 8 controlled layers:

1. **Same road network** — identical .net.xml file for all strategies
2. **Same traffic demand** — identical VPLPH, route generation, random seeds
3. **Same observations** — 14-feature intersection-agnostic vector (lane occupancy, halted occupancy, speed ratio, phase ratios, network rankings)
4. **Same actions** — binary phase switching with identical timing constraints
5. **Same reward** — r = -(occupancy + halted_occupancy)^2
6. **Same algorithm** — PPO with identical hyperparameters and network architecture
7. **Same training budget** — 25 episodes per configuration
8. **Same evaluation** — Monte Carlo runs with identical seeds, metrics from SUMO tripinfo

## Topologies

| Network | Intersections | Lanes | Demand |
|---------|--------------|-------|--------|
| Grid 3x3 | 9 | 24 | 360 VPLPH |
| Grid 5x5 | 25 | heterogeneous | 360 VPLPH |

Synthetic grids with heterogeneous lane counts (center intersections have more lanes than edges). Standard in the FedRL traffic literature (Ye et al. 2021, Hudson et al. 2022, Fu et al. 2025).

## Quick Start

### Prerequisites

- Python 3.10+
- Node.js 18+
- SUMO 1.x (required for real experiments; mock mode available without it)

### Backend

```bash
cd BackEnd
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Run Experiments

```bash
# Train all strategies with incremental saving (resume-safe)
cd BackEnd && python scripts/run_all_training.py --episodes 25 --eval-runs 5

# Resume after interruption
cd BackEnd && python scripts/run_all_training.py --episodes 25 --eval-runs 5 --resume

# Generate all figures
cd BackEnd && python scripts/generate_all_figures.py
```

### Frontend (Dashboard)

```bash
cd FrontEnd
npm install
npx vite --port 5173
```

## Project Structure

```
BackEnd/
├── seal/                    # SEAL RL framework
│   ├── trainer/
│   │   ├── fed_agent.py     # FedRL trainer with FedAvg aggregation
│   │   ├── multi_agent.py   # MARL trainer
│   │   ├── single_agent.py  # SARL trainer
│   │   ├── fedprox_policy.py# FedProx PPO policy (extension)
│   │   └── weight_aggr.py   # Aggregation weight functions
│   └── sumo/
│       ├── env.py           # SUMO environment with observations/rewards
│       └── kernel/          # TraCI wrappers for traffic lights
├── configs/SMARTCOMP/       # SUMO .net.xml network files
├── scripts/
│   ├── run_all_training.py  # Experiment campaign runner
│   ├── run_campaign.py      # Baseline evaluation runner
│   └── generate_all_figures.py # Figure generation
├── results/
│   ├── campaigns/           # Raw experiment results (JSON)
│   └── figures/             # Generated charts and tables
└── api/                     # FastAPI application
    ├── evaluation/          # Monte Carlo evaluation pipeline
    └── training_runner.py   # Trainer factory

FrontEnd/
└── src/
    ├── components/SimCanvas.tsx  # Canvas2D traffic renderer
    ├── hooks/                    # WebSocket streaming hooks
    └── pages/                    # Dashboard pages
```

## References

- Ye et al. (2021) — FedLight: Federated RL for Traffic Signal Control. DAC.
- Hudson et al. (2022) — SEAL: Smart Edge-Enabled Traffic Light Control. SMARTCOMP.
- Bao et al. (2023) — Scalable FL for Traffic Signal Control. Scientific Reports.
- Li et al. (2020) — Federated Optimization in Heterogeneous Networks (FedProx). MLSys.
- Ault & Sharon (2021) — RESCO: RL Benchmarks for Traffic Signal Control. NeurIPS.
- McMahan et al. (2017) — Communication-Efficient Learning of Deep Networks (FedAvg). AISTATS.
