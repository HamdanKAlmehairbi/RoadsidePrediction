# Federated Graph-Based Traffic Congestion Forecasting

## Project Overview
A federated learning system for **graph-based spatio-temporal traffic congestion forecasting** using:
- **STREETS dataset**: 320 cameras across 2 communities (Buffalo Grove + Gurnee, IL)
- **Graph models**: T-GCN and AGCRN for spatio-temporal forecasting on camera network graph
- **CameraCNN**: MobileNetV2-based perception layer (images → congestion estimates)
- **Federated learning**: Subgraph-based training — each client trains on its own camera subgraph
- **Output**: Congestion score [0.0, 1.0] forecasted at 15-min horizon

## Architecture Pipeline
```
Stage 1: CameraCNN (MobileNetV2) processes camera images → congestion estimate per camera
Stage 2: Stack per-camera estimates over time → (N_nodes, T, F) time-series
Stage 3: Graph construction (STREETS directed graph + optional learned adaptive adjacency)
Stage 4: Spatio-temporal model (T-GCN or AGCRN) → forecast next 3 steps (15 min)
Stage 5: Federated learning via SUBGRAPH training — each client trains on own community
```

## Current Status
- [x] STREETS data pipeline (graph, counts, state labels, images)
- [x] Camera CNN model (MobileNetV2 perception layer)
- [x] T-GCN model (GCN + GRU, ~17K params)
- [x] AGCRN model (adaptive learned graph + node-adaptive, ~398K params)
- [x] Baseline models (HA, Persistence, Per-Node LSTM, VAR)
- [x] Graph dataset + collate utilities
- [x] Federated learning infrastructure (FedAvg + GraphClient subgraph training)
- [x] Centralized + federated graph trainers (with cosine LR scheduler)
- [x] GraphFusionModel (CNN → graph pipeline)
- [x] Experiment scripts (centralized, federated, baselines, vision, ablation)
- [x] Full test suite — 84 tests passing
- [x] Baseline experiment results (HA, Persistence, LSTM, VAR)
- [x] Centralized experiment results (T-GCN, AGCRN — 30 epochs, input_steps=3)
- [ ] Centralized v2 results (100 epochs, input_steps=12, cosine scheduler, single community)
- [ ] Federated experiment results
- [ ] Vision pipeline results (requires STREETS images)

## Experiment Results So Far

### Baselines (input_steps=3, forecast_steps=3, 15-min horizon)
| Model | Params | MAE | RMSE | R² | Acc@10% |
|-------|--------|-----|------|-----|---------|
| Historical Average | 0 | **0.0404** | 0.0715 | 0.638 | 87.9% |
| Persistence | 0 | 0.0437 | 0.0816 | 0.529 | 86.5% |
| Per-Node LSTM | 4.6K | 0.0433 | **0.0701** | **0.652** | **88.3%** |
| VAR | 922K | 0.0734 | 0.1037 | 0.238 | 74.4% |

### Centralized v1 (input_steps=3, 30 epochs, no scheduler)
| Model | Params | MAE | RMSE | R² | Acc@10% |
|-------|--------|-----|------|-----|---------|
| T-GCN (full 320 nodes) | 17K | 0.0503 | 0.0752 | 0.600 | 86.5% |
| AGCRN (full 320 nodes) | 398K | 0.0489 | 0.0732 | 0.621 | 87.1% |

### Analysis
- Graph models underperform HA and LSTM with only 3-step input window
- AGCRN > T-GCN (learned adjacency helps, but gap is small)
- VAR catastrophically overfits (922K params for 320-node linear model)
- High MAPE across all models due to near-zero congestion values (most cameras free-flow)
- **Root cause**: input_steps=3 (15 min history) is too short for graph models to learn spatial propagation

### Planned Improvements (centralized v2, in progress)
- input_steps=12 (1 hour history) — more temporal context
- 100 epochs with cosine LR scheduler and patience=20
- Single community training (161 nodes) to reduce graph size
- Initial run showed ~30s/epoch for T-GCN on 161 nodes with 12-step input

## Project Structure
```
RoadsidePrediction/
├── CLAUDE.md                    # This file
├── requirements.txt             # Dependencies (Python 3.9-3.12, PyTorch, pandas)
├── configs/
│   └── default_config.yaml      # STREETS + graph + FL settings
│
├── data/                        # Data pipeline
│   ├── __init__.py
│   ├── streets_loader.py        # STREETS dataset loader (graph, counts, states, images)
│   ├── graph_dataset.py         # GraphTimeSeriesDataset + collate_fn
│   └── camera_dataset.py        # PyTorch Dataset for camera images
│
├── models/                      # Model architectures
│   ├── __init__.py
│   ├── camera_cnn.py            # CameraCNN (MobileNetV2 perception layer)
│   ├── graph_forecaster.py      # T-GCN (GCN inside GRU gates)
│   ├── agcrn.py                 # AGCRN (adaptive graph + node-adaptive weights)
│   ├── adaptive_graph.py        # AdaptiveAdjacency (learned graph for T-GCN)
│   ├── baselines.py             # HA, Persistence, PerNodeLSTM, VARModel
│   └── fusion_model.py          # FusionModel, LateFusionModel, GraphFusionModel
│
├── federated/                   # FL infrastructure
│   ├── __init__.py
│   ├── client.py                # Base FLClient ABC
│   ├── camera_client.py         # CameraClient (roadside cameras)
│   ├── graph_client.py          # GraphClient (subgraph-based FL)
│   ├── server.py                # FLServer with graph evaluation
│   └── fedavg.py                # FedAvg aggregation
│
├── training/                    # Training logic
│   ├── __init__.py
│   ├── centralized_trainer.py   # Centralized multimodal training
│   ├── federated_trainer.py     # Federated training orchestration
│   ├── graph_trainer.py         # GraphCentralizedTrainer (cosine scheduler support)
│   ├── graph_federated_trainer.py # GraphFederatedTrainer (subgraph FL)
│   └── metrics.py               # MSE, MAE, R², MAPE, per-horizon, per-node metrics
│
├── experiments/                 # Experiment scripts
│   ├── run_streets_centralized.py  # Centralized T-GCN + AGCRN (single community + full)
│   ├── run_streets_federated.py    # Federated T-GCN + AGCRN (10-20 clients)
│   ├── run_streets_baselines.py    # HA, persistence, LSTM, VAR
│   ├── run_streets_vision.py       # CameraCNN on STREETS images
│   └── run_streets_ablation.py     # Full ablation (9 experiments)
│
├── utils/                       # Utilities
│   ├── logger.py                # Logging utilities
│   └── visualization.py         # Plotting
│
├── tests/                       # Unit tests (84 tests, all passing)
│   ├── test_graph_models.py     # T-GCN, AGCRN, AdaptiveAdjacency, GraphDataset
│   ├── test_baselines.py        # Baselines + new metrics
│   ├── test_streets.py          # STREETS data loading (synthetic data)
│   └── test_graph_federated.py  # GraphClient, subgraph isolation, FL round
│
├── results/                     # Experiment outputs
│   ├── baseline_results.json    # HA, Persistence, LSTM, VAR
│   └── centralized_results.json # T-GCN, AGCRN (v1, 30 epochs)
│
└── STREETS/                     # Dataset (not in git)
    ├── graphs/
    │   ├── buffalogrove/buffalogrove-graph.json
    │   └── gurnee/gurnee-graph.json
    ├── trafficcounts/
    │   ├── 2018/*.json           # 31 daily files (10-min intervals)
    │   └── 2019/*.json           # 44 daily files (5-min intervals)
    ├── trafficstate/
    │   ├── traffic_state_labels.json
    │   └── *.jpg                 # 6400 annotated images
    └── 2019-7-3_2019-7-9/        # Weekly image archives (optional)
```

## Dataset: STREETS (NeurIPS 2019)

### Overview
- **320 cameras** (161 Buffalo Grove + 159 Gurnee) = 640 sensors (inbound/outbound per camera)
- **2 directed-graph communities** with pre-built adjacency matrices
- **2.5 months of data**: 2018 Aug-Sep (10-min), 2019 Jun-Jul (5-min)
- **Vehicle counts** at each camera approach (inbound + outbound)
- **Traffic state labels**: free-flow ('f') / blocked ('b') for ~6400 images
- **Source**: https://databank.illinois.edu/datasets/IDB-3671567

### Dataset Stats (from real data)
- 320 cameras total (161 BG + 159 GU)
- 79,789 unique timestamps across both years
- Mean vehicle count: 6.10, max: 69
- 3.4% blocked traffic state labels
- Train (2018): 32,772 timestamps
- Test (2019): 47,017 timestamps

### Data Format
```
graphs/: Per-community JSON with sensor-dictionary, adjacency-matrix, distance-matrix
trafficcounts/: Daily JSON files mapping camera → {image: {inbound, outbound, timestamp}}
trafficstate/: traffic_state_labels.json with 'f'/'b' binary labels
```

### Congestion Score
```
score = count / max_count, clipped to [0, 1]
Calibrated by state labels:
  blocked (state='b'): score = max(score, 0.6)
  free-flow (state='f'): score = min(score, 0.5)
```

## Key Interfaces

### Data Pipeline
```python
from data.streets_loader import (
    load_streets_graph,              # → adjacency, communities, camera_ids
    load_streets_traffic_counts,     # → DataFrame (timestamps × cameras)
    load_streets_traffic_state,      # → DataFrame (0=free, 1=blocked)
    compute_congestion_from_counts,  # → DataFrame [0.0, 1.0]
    build_node_timeseries,           # → ndarray (T, N, 1)
    split_by_year,                   # → chronological train/test split
    partition_by_camera_subgroups,   # → client → camera_ids mapping
)

from data.graph_dataset import GraphTimeSeriesDataset, graph_collate_fn
```

### Models
```python
# Graph forecasting (spatio-temporal)
class TGCN(nn.Module):
    def forward(self, x, adj) -> (B, N, T_out)
    def forward_predict(self, x, adj) -> (B, N, T_out)  # sigmoid bounded

class AGCRN(nn.Module):  # Learned graph, no predefined adj needed
    def forward(self, x, adj=None) -> (B, N, T_out)
    def forward_predict(self, x, adj=None) -> (B, N, T_out)

# Camera perception
class CameraCNN(nn.Module):
    def forward(self, x) -> (batch, feature_dim)
    def forward_predict(self, x) -> (batch, 1) in [0,1]

# Baselines
class HistoricalAverage   # 0 params, mean of input
class PersistenceModel    # 0 params, last value
class PerNodeLSTM         # Shared LSTM, no graph structure
class VARModel            # Linear multivariate (classical)
```

### Federated Learning
```python
class GraphClient(FLClient):
    # Trains on LOCAL SUBGRAPH only — no access to other clients' data
    def train_local(self, epochs) -> Dict[str, Tensor]
    def set_model_params(self, params)  # Only loads shape-compatible weights

class FLServer:
    def broadcast() -> Dict[str, Tensor]
    def aggregate(updates, weights)
    def evaluate_global_graph(test_loader, adjacency) -> Dict
```

## Configuration
```yaml
data:
  dataset: "streets"
  streets:
    dataroot: "./STREETS"
    train_year: 2018         # chronological split
    test_year: 2019
    max_count_normalization: 50

graph:
  model: "agcrn"             # "tgcn" or "agcrn"
  hidden_dim: 64
  input_steps: 12            # 1 hour input (at 5-min intervals)
  forecast_steps: 3          # 15 min forecast

federated:
  num_graph_clients: 10      # 10-20 for heterogeneity analysis
  rounds: 50
  local_epochs: 5
```

## Running the Project

```bash
# Setup (Python 3.11 recommended)
conda create -n streets python=3.11
conda activate streets
pip install -r requirements.txt

# IMPORTANT: Run from project root with PYTHONPATH
PYTHONPATH=. python experiments/run_streets_baselines.py
PYTHONPATH=. python experiments/run_streets_centralized.py
PYTHONPATH=. python experiments/run_streets_federated.py
PYTHONPATH=. python experiments/run_streets_ablation.py
PYTHONPATH=. python experiments/run_streets_vision.py    # requires image data

# Run tests
python -m pytest tests/ -v   # 84 tests, all passing
```

## Performance Notes
- T-GCN on 161 nodes (12-step input): ~27-30s/epoch on CPU
- AGCRN on 320 nodes (3-step input): ~65s/epoch on CPU
- AGCRN on 161 nodes (12-step input): may be slow, consider hidden_dim=32
- Baseline experiment completes in ~5 minutes total
- Python 3.11 at `/Library/Frameworks/Python.framework/Versions/3.11/bin/python3.11`

## Notes
- Python 3.9-3.12 required (PyTorch not compatible with 3.13)
- STREETS dataset (~330 MB counts-only, 188 GB with images)
- No torch-geometric dependency — GCN implemented as matrix multiply
- FL privacy: only model weights shared, never features or raw data
- Must use `PYTHONPATH=.` when running experiment scripts
