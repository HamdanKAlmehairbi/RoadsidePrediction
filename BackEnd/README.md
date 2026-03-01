# SEAL Dashboard Backend

FastAPI backend for the SEAL (Smart Edge-enabled Adaptive Lights) traffic simulation dashboard.

## Prerequisites

### SUMO (optional)

SUMO is required for live traffic simulation. If not installed, the backend runs in mock mode with synthetic vehicle data.

**Windows:**
Download the installer from https://sumo.dlr.de/docs/Installing/index.html#windows

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

After installation, ensure `sumo` is on your PATH:
```bash
sumo --version
```

Set `SUMO_HOME` if needed:
```bash
export SUMO_HOME=/usr/share/sumo  # Linux
export SUMO_HOME=/opt/homebrew/share/sumo  # macOS (Homebrew)
```

### Python Dependencies

```bash
cd BackEnd
pip install -r requirements.txt
```

## Starting the Server

```bash
cd BackEnd
uvicorn api.main:app --reload --port 8000
```

The API is available at `http://localhost:8000`.

## Mock Mode

If SUMO is not installed, the backend automatically falls back to mock mode:
- Simulation endpoints generate synthetic vehicle movements and traffic light states
- Training endpoints generate synthetic reward curves that improve over episodes
- All REST and WebSocket endpoints function identically

A warning is logged at startup: `TraCI not available -- simulation will use mock mode`

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/networks` | List available topologies |
| GET | `/api/network/{topology}` | Get network node/edge layout |
| GET | `/api/weights` | List pre-trained weight files |
| POST | `/api/simulate` | Start a simulation job |
| POST | `/api/train` | Start a training job |
| GET | `/api/results/{job_id}` | Get job results |
| WS | `/ws/simulate/{job_id}` | Stream simulation frames |
| WS | `/ws/train/{job_id}` | Stream training episodes |
