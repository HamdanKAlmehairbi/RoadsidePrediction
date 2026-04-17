# Benchmarking Backend

FastAPI backend for the FedRL traffic signal control benchmarking framework. Runs SUMO-based RL experiments comparing SARL, MARL, and FedRL training strategies under identical conditions.

## Prerequisites

### SUMO (required for experiments)

SUMO is required for training and evaluation. Without it, the backend runs in mock mode with synthetic data (sufficient for dashboard development only).

**Windows:**
Download from https://sumo.dlr.de/docs/Installing/index.html#windows

**Linux (Ubuntu/Debian):**
```bash
sudo add-apt-repository ppa:sumo/stable
sudo apt update
sudo apt install sumo sumo-tools
```

**macOS:**
```bash
brew install sumo
```

Set `SUMO_HOME` if needed:
```bash
export SUMO_HOME=/usr/share/sumo  # Linux
export SUMO_HOME=/opt/homebrew/share/sumo  # macOS
```

### Python Dependencies

```bash
cd BackEnd
pip install -r requirements.txt
```

## Running Experiments

### Full Training Campaign (all strategies)

```bash
# Train SARL, MARL, FedRL on both topologies + FedProx + ToD ablations
python scripts/run_all_training.py --episodes 25 --eval-runs 5

# Resume after interruption (skips completed configs)
python scripts/run_all_training.py --episodes 25 --eval-runs 5 --resume
```

Results save incrementally after each config to `results/campaigns/training-curves/results.json`.

### Baseline Evaluation (pre-trained weights only)

```bash
python scripts/run_campaign.py --campaign baseline --n-eval-runs 10
```

### Generate Figures

```bash
python scripts/generate_all_figures.py
```

Outputs to `results/figures/`.

## Starting the Dashboard Server

```bash
uvicorn api.main:app --reload --port 8000
```

## Benchmarking Framework

The evaluation framework controls all variables except the training strategy:

| Layer | What's Controlled | How |
|-------|-------------------|-----|
| Simulator | SUMO 1.26.0, same physics | Same binary |
| Network | Same .net.xml | Loaded once, never modified |
| Demand | 360 VPLPH, same seeds | Same randomTrips call |
| Observations | 14 features, normalized [0,1] | Same `get_observation()` |
| Reward | r = -(o + h)^2 | Same code path |
| Algorithm | PPO, same hyperparameters | Same RLlib config |
| Training | Same episode count | No early stopping |
| Evaluation | Same MC seeds (42-46) | Same `run_monte_carlo()` |

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/networks` | List available topologies |
| GET | `/api/network/{topology}` | Get network node/edge layout |
| GET | `/api/weights` | List trained weight files |
| POST | `/api/simulate` | Start a simulation job |
| POST | `/api/train` | Start a training job |
| GET | `/api/results/{job_id}` | Get job results |
| WS | `/ws/simulate/{job_id}` | Stream simulation frames |
| WS | `/ws/train/{job_id}` | Stream training episodes |

## Topologies

- `grid-3x3` — 9 signalized intersections, 24 controlled lanes
- `grid-5x5` — 25 signalized intersections, heterogeneous lane counts
- `manhattan` — 21 intersections from OpenStreetMap (experimental, observation layer needs adaptation)
- `cologne-8` — 8 real-world intersections from RESCO benchmark (experimental)
