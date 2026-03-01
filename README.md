# SEAL Dashboard

A real-time traffic simulation dashboard for the **SEAL** (Smart Edge-Assisted Learning) federated reinforcement learning framework. Visualizes AI-controlled traffic lights learning to optimize traffic flow across SUMO grid networks.

## What It Does

SUMO simulates a city grid with cars and traffic lights. This project replaces fixed-timer traffic lights with AI agents that learn optimal signal timing through reinforcement learning. Three approaches are compared:

- **SARL** — Single shared model for all intersections
- **MARL** — Independent model per intersection, no sharing
- **FedRL** — Independent models that periodically share knowledge via an edge server (36% less communication than MARL, ~2% traffic performance trade-off)

The dashboard streams live simulation state (vehicle positions, speeds, traffic light phases) from SUMO via WebSocket and renders an animated bird's-eye view on an HTML Canvas.

## Architecture

```
FrontEnd/          React + Vite + shadcn/ui + Tailwind + Recharts + Canvas API
    │  REST + WebSocket
BackEnd/           FastAPI + uvicorn (SEAL framework copied in)
    │  TraCI (Python API)
SUMO               installed on host machine separately
```

## Pages

| Route | Description |
|-------|-------------|
| `/` | Dashboard overview with stats and how-it-works |
| `/simulation` | Live simulation canvas with vehicle animation |
| `/compare` | Side-by-side policy comparison (e.g., FedRL vs MARL) |
| `/training` | Train agents and watch reward curves in real time |
| `/communication` | Communication cost analysis charts and tables |

## Quick Start

### Prerequisites

- Python 3.9+
- Node.js 18+
- SUMO (optional — runs in mock mode without it)

### Backend

```bash
cd BackEnd
pip install -r requirements.txt
python -m uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

### Frontend

```bash
cd FrontEnd
npm install
npx vite --port 5173
```

Open `http://localhost:5173` in your browser.

### Installing SUMO (optional)

Without SUMO installed, the backend runs in **mock mode** with simulated vehicle data. To use real simulations:

1. Install SUMO from [eclipse.dev/sumo](https://eclipse.dev/sumo/)
2. Set the `SUMO_HOME` environment variable
3. Restart the backend — TraCI will be detected automatically

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/networks` | GET | List available topologies |
| `/api/network/{topology}` | GET | Road layout (nodes, edges, bounds) |
| `/api/weights` | GET | List trained weight files |
| `/api/simulate` | POST | Start a simulation job |
| `/api/train` | POST | Start a training job |
| `/api/results/{job_id}` | GET | Get job results |
| `/ws/simulate/{job_id}` | WS | Stream vehicle positions + traffic light states |
| `/ws/train/{job_id}` | WS | Stream episode reward updates |

## Network Topologies

Three SUMO grid networks are included:

- `grid-3x3` — 9 intersections (400x400)
- `grid-5x5` — 25 intersections (600x600)
- `grid-7x7` — 49 intersections (800x800)

## Project Structure

```
BackEnd/
├── seal/                    # SEAL RL framework (copied from SUMO-FedRL-main/)
├── configs/SMARTCOMP/       # SUMO .net.xml network files
├── example_weights/         # Pre-trained .pkl weight files
└── api/                     # FastAPI application
    ├── main.py              # App setup, CORS, router mounting
    ├── jobs.py              # In-memory job store
    ├── routes/              # REST endpoints
    └── ws/                  # WebSocket endpoints

FrontEnd/
└── src/
    ├── components/
    │   ├── SimCanvas.tsx    # Canvas2D renderer (roads, vehicles, signals)
    │   └── ui/              # shadcn/ui primitives
    ├── hooks/
    │   ├── useSimStream.ts  # WebSocket hook for simulation frames
    │   └── useTrainStream.ts# WebSocket hook for training episodes
    ├── pages/               # 5 route pages
    └── lib/api.ts           # API client functions
```
