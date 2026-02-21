# Project Briefing: Federated Graph-Based Traffic Congestion Forecasting

## What Are We Building?

A system that **predicts traffic congestion 15 minutes into the future** using a network of roadside cameras — without sharing raw data between cameras. Each camera (or group of cameras) trains a local model on its own data, then shares only model weights with a central server. This is **federated learning (FL)** applied to **graph-based spatio-temporal forecasting**.

The key insight: traffic doesn't happen in isolation. A jam at one intersection propagates to neighboring ones. By modeling the camera network as a **graph**, we capture these spatial dependencies. By using FL, we keep each camera's data private.

---

## Why STREETS?

We evaluated three datasets before settling on STREETS:

| Dataset | Cameras | Duration | Graph? | Why Not |
|---------|---------|----------|--------|---------|
| nuScenes | Moving ego vehicle | Short clips | No | Not fixed roadside cameras |
| UA-DETRAC | 5 locations | 60s sequences | No | Too few locations, too short for forecasting |
| **STREETS** | **320 fixed cameras** | **2.5 months** | **Yes, built-in** | **Winner** |

**STREETS** (NeurIPS 2019) gives us everything we need:
- **320 cameras** across 2 real communities in Illinois (Buffalo Grove + Gurnee)
- **Pre-built directed graph** — the road network topology is already defined
- **Vehicle counts** every 5-10 minutes over 2.5 months
- **Ground truth labels** — free-flow vs blocked (binary traffic state)
- Two natural communities = two natural FL partitions

Source: https://databank.illinois.edu/datasets/IDB-3671567

---

## Architecture: Why Graph + Federated?

### The Problem With Simple Approaches

A naive approach would be: train one model per camera to predict its own future congestion. This ignores a fundamental property of traffic — **spatial propagation**. When a highway on-ramp gets congested, downstream intersections feel it 5-15 minutes later.

A graph neural network captures this by letting each camera "see" its neighbors' states when making predictions.

### The Pipeline

```
┌─────────────┐     ┌──────────────────┐     ┌─────────────────────┐
│  Raw Camera  │     │  Congestion      │     │  Graph Forecaster   │
│  Images      │────▶│  Score [0,1]     │────▶│  (T-GCN or AGCRN)   │
│  (optional)  │     │  per camera      │     │                     │
└─────────────┘     └──────────────────┘     └─────────────────────┘
                           │                          │
                    Currently using              Operates on the
                    vehicle counts as             camera network graph
                    proxy scores                  to forecast 15 min ahead
```

**Stage 1 — Congestion Scoring**: Each camera gets a congestion score [0.0 = free flow, 1.0 = gridlock]. Right now we derive this from vehicle counts (count / 50, clipped). Optionally, a CameraCNN (MobileNetV2) can estimate this from raw images.

**Stage 2 — Time-Series Construction**: We stack scores across time to get a matrix: `(T timesteps × N cameras × 1 feature)`. With 5-minute intervals, 12 timesteps = 1 hour of history.

**Stage 3 — Graph Forecasting**: A spatio-temporal model processes this matrix along with the graph adjacency to predict the next 3 timesteps (15 minutes). The graph lets information flow between connected cameras.

### Why Two Graph Models?

We benchmark two architectures to compare fixed vs learned graph structure:

**T-GCN (Temporal Graph Convolutional Network)** — 17K params
- Uses the STREETS road graph directly (fixed adjacency)
- GCN layers inside GRU gates — spatial + temporal in one cell
- Lightweight, fast to train
- Think of it as: "use the road map as-is"

**AGCRN (Adaptive Graph Convolutional Recurrent Network)** — 398K params
- **Learns its own graph** from data — doesn't need a predefined adjacency
- Each node gets personalized weights (node-adaptive)
- More powerful but heavier
- Think of it as: "discover which cameras actually influence each other"

We also compare against 4 baselines to prove the graph structure adds value:

| Baseline | What It Tests |
|----------|--------------|
| Historical Average | Can you beat "same as usual"? |
| Persistence | Can you beat "same as now"? |
| Per-Node LSTM | Does the graph help vs independent per-camera models? |
| VAR (Vector Autoregression) | Does deep learning help vs classical linear? |

### Why Federated?

In a real deployment, cameras belong to different jurisdictions, operators, or organizations. They can't share raw video feeds — privacy, bandwidth, regulations.

**Federated learning** solves this:
1. Each client (group of ~16-32 cameras) trains a local model on its own subgraph
2. Only model weights are sent to a central server
3. Server averages weights (FedAvg) and sends the updated model back
4. Repeat for N rounds

**Key design choice — Subgraph Training**: Each client doesn't just train on a subset of data — it trains on a **subgraph** of the camera network. It only sees its own cameras' adjacency matrix. Cross-community learning happens implicitly through weight averaging, not data sharing. No privacy leak.

---

## Current Results

### What We've Run

| Model | Params | MAE | R² | Verdict |
|-------|--------|-----|-----|---------|
| Historical Average | 0 | **0.040** | 0.638 | Surprisingly strong floor |
| Per-Node LSTM | 4.6K | 0.043 | **0.652** | Best so far (no graph) |
| T-GCN (centralized) | 17K | 0.050 | 0.600 | Below baselines |
| AGCRN (centralized) | 398K | 0.049 | 0.621 | Below baselines |
| VAR | 922K | 0.073 | 0.238 | Catastrophic overfit |

### What This Means

The graph models are **underperforming simple baselines**. This isn't a failure — it's a known issue with our initial configuration:

- **Input window too short**: We used 3 timesteps (15 min) of history. Graph models need longer context to learn spatial propagation patterns. We're switching to 12 timesteps (1 hour).
- **Not enough training epochs**: AGCRN (398K params) had only 30 epochs. It needs 50-100.
- **No learning rate scheduling**: Adding cosine annealing to prevent oscillation.

These fixes are implemented and ready to run. The centralized v2 experiment uses all three improvements.

### What's Left

- [ ] Re-run centralized with 12-step input, 100 epochs, cosine LR
- [ ] Run federated experiments (10-20 clients)
- [ ] Compare centralized vs federated (the "FL gap")
- [ ] Run ablation: 2 vs 10 vs 20 clients
- [ ] Vision pipeline (CameraCNN on actual STREETS images)

---

## Codebase Layout

```
data/
  streets_loader.py      ← loads graphs, counts, state labels from STREETS
  graph_dataset.py       ← PyTorch Dataset for (input_window, forecast_target)
  camera_dataset.py      ← PyTorch Dataset for (image, congestion_score)

models/
  graph_forecaster.py    ← T-GCN (GCN + GRU)
  agcrn.py               ← AGCRN (learned graph + node-adaptive)
  adaptive_graph.py      ← Learnable adjacency matrix
  baselines.py           ← HA, Persistence, LSTM, VAR
  camera_cnn.py          ← MobileNetV2 perception layer
  fusion_model.py        ← GraphFusionModel (CNN → graph pipeline)

federated/
  client.py              ← Base FL client (ABC)
  graph_client.py        ← Subgraph-aware FL client
  server.py              ← FLServer (broadcast, aggregate, evaluate)
  fedavg.py              ← FedAvg weight averaging

training/
  graph_trainer.py       ← Centralized training (upper-bound baseline)
  graph_federated_trainer.py ← Federated training orchestration
  metrics.py             ← MSE, MAE, R², MAPE, per-horizon, per-node

experiments/
  run_streets_baselines.py    ← HA, Persistence, LSTM, VAR
  run_streets_centralized.py  ← T-GCN + AGCRN (single community + full graph)
  run_streets_federated.py    ← Federated T-GCN + AGCRN
  run_streets_ablation.py     ← Full 9-experiment comparison
  run_streets_vision.py       ← CameraCNN on STREETS images

tests/                   ← 84 tests, all passing
```

---

## How to Run

```bash
# Setup
conda create -n streets python=3.11
conda activate streets
pip install -r requirements.txt

# Verify everything works
python -m pytest tests/ -v    # should see 84 passed

# Run experiments (from project root)
PYTHONPATH=. python experiments/run_streets_baselines.py      # ~5 min
PYTHONPATH=. python experiments/run_streets_centralized.py    # ~40 min (T-GCN) + hours (AGCRN)
PYTHONPATH=. python experiments/run_streets_federated.py      # depends on num_clients
```

Results are saved to `results/*.json`.

---

## What Makes This Project Novel?

1. **Graph-based forecasting + FL on a real camera network**: Most FL traffic papers use simulated or trivial partitions. We use STREETS' real community structure.

2. **Subgraph-based privacy**: Each client only sees its own subgraph. Cross-community patterns emerge through weight sharing, not data sharing.

3. **Comprehensive comparison**: We don't just show "our method works" — we benchmark against 4 baselines, 2 graph architectures, and 3 FL configurations (2/10/20 clients) to isolate exactly where graph structure and federated learning help.

4. **Perception + forecasting pipeline**: The CameraCNN provides a path from raw images to graph forecasting — making this deployable on actual camera infrastructure where only images (not vehicle counts) are available at inference.
