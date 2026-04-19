# SEAL + ATLAS: RL Training Strategy Benchmarking and Live Comparison for Multi-Intersection Traffic Signal Control

This repository combines two layers: **SEAL**, a standardized benchmarking framework that evaluates 10 RL training strategies for multi-intersection traffic signal control under identical conditions (same SUMO environment, same PPO hyperparameters, same seeds); and **ATLAS**, a live web-based comparison demo that runs two SUMO simulations side-by-side in real time, each driven by a user-selected strategy and topology. The only variable in SEAL is the training strategy -- from fully independent agents to fully shared policies. ATLAS makes those trained strategies interactively comparable through a browser UI backed by subprocess-isolated TraCI sessions.

---

## Demo

![ATLAS live comparison demo](docs/demo.png)

*Replace `docs/demo.png` with an actual screenshot once the demo is running.*

---

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

Pre-trained weights for all 8 RL strategies across three topologies live at:

```
BackEnd/example_weights/ICCPS/Final/{Trainer}/{topology}/ranked.pkl
```

Trainers: `CTDE`, `FedDistill`, `FedRL`, `Gossip`, `HierFed`, `MARL`, `MeanField`, `SARL`
Topologies: `grid-3x3`, `grid-5x5`, `cologne-8`

---

## ATLAS Live Comparison Demo

ATLAS lets you pick any two (topology, strategy) pairs and watch them race in real time through a browser UI. Vehicles animate on an SVG grid, traffic lights tick through phases, and per-intersection metrics update every step.

### Architecture

- **Subprocess isolation** -- `POST /compare/start` spawns one `multiprocessing.Process` per simulation. Each process owns a fresh TraCI socket connection, so the two SUMO instances never share state.
- **Ray-bypassed inference** -- weights are loaded directly with PyTorch (`policy_loader.py`); Ray is not used at runtime. `sim_worker.py` runs the step loop, `dual_runner.py` coordinates both processes.
- **SVG rendering** -- `NetworkCanvas.tsx` renders junctions and vehicles as SVG elements. Junction pixel coordinates are derived from SUMO bounding-box extents (junction-bbox normalization).
- **Random trip generation** -- each simulation session calls SUMO `randomTrips.py` to generate fresh demand, ensuring variety across demo runs.
- **WebSocket streaming** -- the backend streams `{type: "frame", sim_a, sim_b}` messages at the requested `step_hz`. The frontend hook `useCompareSession.ts` drives React state from the stream.

### Key Endpoints

| Method | Path | Payload / Response |
|--------|------|--------------------|
| `POST` | `/compare/start` | `{topology_a, strategy_a, topology_b, strategy_b, horizon, step_hz}` -> `{session_id, ws_url}` |
| `WS` | `/ws/compare/{session_id}` | streams `{type:"frame", sim_a, sim_b}` at `step_hz` |

### Running Locally

```bash
# Backend (Python 3.12, SUMO_HOME set, deps installed — see SEAL Prerequisites below)
cd BackEnd
python -m uvicorn api.main:app --port 8000

# Frontend
cd FrontEnd
npm install
npm run dev   # http://localhost:5173
```

---

## SEAL Framework Usage

### Prerequisites

- Python 3.12 (Ray does not support 3.13)
- SUMO >= 1.20.0
- Node.js 18+ (frontend only)

```bash
cd BackEnd
pip install -r requirements.txt
```

### Train

```bash
# Train specific strategies on specific topologies (resumable, parallel GPU support)
python scripts/train_missing_trainers.py \
    --trainers Gossip HierFed FedRL \
    --topologies grid-3x3 grid-5x5 \
    --num-workers 8
```

### Evaluate

```bash
# Run evaluation campaign (incremental saves, resumable)
python scripts/run_campaign.py \
    --campaign baseline \
    --topologies grid-3x3 grid-5x5 cologne-8 \
    --n-eval-runs 10 \
    --output-name baseline_full
```

### HPC Sweep (4 GPUs, ~12 hours)

```bash
cd BackEnd
bash hpc/run_all.sh
```

Runs 7 phases: grid-7x7 training, cologne-8 extended training (200 episodes), demand sweep, fed_step / alpha / fedprox ablations. See `hpc/README.md`.

### Ablations

```bash
python hpc/run_ablation.py --ablation demand
python hpc/run_ablation.py --ablation all
```

---

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
- **All RL crushes baselines on grids**: 5-15x lower waiting time than fixed-time / max-pressure.
- **Real-world networks need more training**: 50 episodes insufficient for cologne-8 irregular geometry.
- **100% throughput everywhere**: Strategies differ in quality (wait time), not capacity.

---

## Project Structure

```
RoadsidePrediction-FinalTouch/
|-- BackEnd/
|   |-- seal/                        # SEAL RL framework
|   |   |-- trainer/
|   |   |   |-- base.py              # Abstract base trainer
|   |   |   |-- single_agent.py      # SARL
|   |   |   |-- multi_agent.py       # MARL
|   |   |   |-- fed_agent.py         # FedRL (FedAvg, star topology)
|   |   |   |-- gossip_agent.py      # Gossip (peer-to-peer mesh)
|   |   |   |-- hierfed_agent.py     # HierFed (two-tier tree)
|   |   |   |-- feddistill_agent.py  # FedDistill (KL distillation)
|   |   |   |-- mean_field_agent.py  # MeanField (augmented obs)
|   |   |   |-- ctde_agent.py        # CTDE (centralized critic)
|   |   |   |-- fedprox_policy.py    # FedProx custom PPO policy
|   |   |   |-- feddistill_policy.py # FedDistill custom PPO policy
|   |   |   `-- weight_aggr.py       # Aggregation functions
|   |   `-- sumo/
|   |       |-- env.py               # Multi-agent SUMO environment
|   |       |-- ctde_env.py          # CTDE env (global state augmentation)
|   |       |-- mean_field_env.py    # MeanField env (neighbor actions)
|   |       `-- kernel/              # TraCI interface layer
|   |-- api/
|   |   |-- main.py                  # FastAPI app entry point
|   |   |-- jobs.py                  # Background job registry
|   |   |-- training_runner.py       # Trainer factory + training loop
|   |   |-- inference/
|   |   |   |-- dual_runner.py       # Coordinates two sim processes
|   |   |   |-- policy_loader.py     # PyTorch weight loading (no Ray)
|   |   |   `-- sim_worker.py        # Per-process SUMO step loop
|   |   |-- routes/
|   |   |   |-- compare.py           # POST /compare/start
|   |   |   |-- evaluate.py
|   |   |   |-- networks.py
|   |   |   |-- results.py
|   |   |   |-- simulate.py
|   |   |   |-- train.py
|   |   |   `-- weights.py
|   |   |-- ws/
|   |   |   |-- compare.py           # WS /ws/compare/{session_id}
|   |   |   |-- evaluate.py
|   |   |   |-- simulate.py
|   |   |   `-- train.py
|   |   |-- evaluation/              # Monte Carlo eval, metrics, campaign config
|   |   `-- baselines/               # fixed-time, max-pressure implementations
|   |-- configs/SMARTCOMP/           # SUMO .net.xml network files
|   |-- scripts/
|   |   |-- train_missing_trainers.py
|   |   |-- run_campaign.py
|   |   |-- generate_tables.py
|   |   |-- run_all_training.py
|   |   |-- run_full_experiments.py
|   |   |-- run_extension_ablation.py
|   |   |-- run_robustness_test.py
|   |   |-- smoke_test_all.py
|   |   |-- rerun_failed_fedprox.py
|   |   |-- hpc_full_campaign.sh
|   |   |-- generate_all_figures.py
|   |   |-- generate_report.py
|   |   |-- generate_seal_figures.py
|   |   |-- generate_visualizations.py
|   |   `-- gen_download_figs.py
|   |-- hpc/
|   |   |-- run_all.sh               # Master HPC script (7 phases)
|   |   `-- run_ablation.py          # Ablation campaign runner
|   |-- example_weights/ICCPS/Final/ # 24 pre-trained .pkl files
|   |   `-- {Trainer}/{topology}/ranked.pkl
|   `-- results/campaigns/           # Experiment results (JSON)
|
`-- FrontEnd/
    `-- src/
        |-- App.tsx
        |-- main.tsx
        |-- globals.css
        |-- vite-env.d.ts
        |-- components/
        |   |-- Header.tsx           # Top navigation / branding
        |   |-- ConfigBar.tsx        # Topology + strategy dropdowns
        |   |-- SimPanel.tsx         # Per-simulation panel wrapper
        |   |-- NetworkCanvas.tsx    # SVG grid + vehicle animation
        |   `-- ComparisonBar.tsx    # Side-by-side metric display
        |-- hooks/
        |   `-- useCompareSession.ts # WebSocket session state hook
        `-- lib/
            |-- types.ts             # Shared TypeScript types
            `-- mockData.ts          # Offline fixture data
```

---

## Controlled Comparison (8 Layers)

Every experiment controls these factors so the training strategy is the only variable:

1. **Same simulator** -- SUMO, same physics
2. **Same network** -- identical .net.xml
3. **Same demand** -- identical VPLPH, same seeds
4. **Same observations** -- 14-feature vector per intersection
5. **Same reward** -- r = -(occupancy + halted_occupancy)^2
6. **Same algorithm** -- PPO (lr=0.001, gamma=0.95)
7. **Same training budget** -- 50 episodes per config
8. **Same evaluation** -- 10 Monte Carlo seeds with bootstrap 95% CI

---

## References

- Ye et al. (2021) -- FedLight: Federated RL for Traffic Signal Control. DAC.
- Hudson et al. (2022) -- SEAL: Smart Edge-Enabled Traffic Light Control. SMARTCOMP.
- Bao et al. (2023) -- Scalable FL for Traffic Signal Control. Scientific Reports.
- Li et al. (2020) -- Federated Optimization in Heterogeneous Networks (FedProx). MLSys.
- Ault & Sharon (2021) -- RESCO: RL Benchmarks for Traffic Signal Control. NeurIPS.
- McMahan et al. (2017) -- Communication-Efficient Learning of Deep Networks (FedAvg). AISTATS.
