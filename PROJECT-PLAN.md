# SEAL Dashboard — Project Plan

> Use with: `/build-with-agent-team PROJECT-PLAN.md 2`

---

## What This System Does (Plain English)

SUMO is a traffic simulator — a city grid with cars driving around, controlled by traffic lights. Normally lights follow a fixed timer. This project replaces that timer with AI.

Each traffic light is an AI agent that watches how many cars are waiting at its lanes, how fast they're moving, and what its neighbours look like — then decides every few seconds whether to switch phase or hold. The AI learns by running thousands of simulation episodes; bad decisions cause congestion and a negative reward signal, good decisions let traffic flow.

Three approaches are compared:
- **SARL** — one shared brain for every intersection. Simple, cheap, misses local differences.
- **MARL** — every intersection has its own brain and never shares. Smart but expensive to run.
- **FedRL** — every intersection has its own brain, but periodically they pool what they've learned via an edge server. The paper's result: 36% less communication than MARL, barely 2% worse traffic performance.

SUMO exposes all simulation state (vehicle positions, speeds, traffic light phases) via its TraCI Python API at every timestep. The backend reads this in real time and streams it to the browser, so the frontend can animate a live bird's-eye view of cars moving through the grid — the most visual and compelling part of the app.

**Hard rule:** `SUMO-FedRL-main/` is never modified. The backend copies what it needs.
**Hard rule:** `LovableOutput/` is never modified. The frontend copies what it needs.

---

## Architecture

```
LovableOutput/     ← Lovable-generated scaffold (read-only source)
    │  copy into
FrontEnd/          ← React + Vite + shadcn/ui + Tailwind + Recharts + Canvas API
    │  REST + WebSocket
BackEnd/           ← FastAPI + uvicorn (copied SEAL framework inside)
    │  TraCI (Python API)
SUMO (system)      ← installed on host machine separately (sumo + sumo-gui)
```

The two WebSocket streams are the core of the live experience:
- `/ws/simulate/{job_id}` — streams vehicle positions + TLS states from a running SUMO simulation
- `/ws/train/{job_id}` — streams episode reward updates from a running training job

---

## Project Structure

```
BackEnd/
├── seal/                            # copied from SUMO-FedRL-main/seal/
├── configs/SMARTCOMP/               # copied .net.xml files (3x3, 5x5, 7x7)
├── example_weights/ICCPS/Final/     # copied .pkl weight files
├── netfiles.py                      # copied
├── requirements.txt                 # copied + fastapi, uvicorn, websockets added
└── api/
    ├── main.py                      # FastAPI app, CORS, router mounting
    ├── jobs.py                      # in-memory job store {job_id: status/results}
    ├── routes/
    │   ├── networks.py              # GET /api/networks, GET /api/network/{topology}
    │   ├── weights.py               # GET /api/weights
    │   ├── simulate.py              # POST /api/simulate
    │   ├── train.py                 # POST /api/train
    │   └── results.py               # GET /api/results/{job_id}
    └── ws/
        ├── simulate.py              # WS /ws/simulate/{job_id} — streams vehicle+TLS state
        └── train.py                 # WS /ws/train/{job_id} — streams episode rewards

FrontEnd/                            # copied from LovableOutput/seal-traffic-flow-main/ then wired up
├── index.html
├── package.json                     # React 18, Vite, shadcn/ui, Recharts, React Router, React Query
├── vite.config.ts
├── tailwind.config.ts               # Dark theme + SEAL accent colours
├── components.json                  # shadcn/ui config
└── src/
    ├── main.tsx
    ├── App.tsx                      # BrowserRouter + QueryClientProvider + Toaster; all 5 routes
    ├── index.css                    # Global CSS vars (#0f1117 bg, #1a1d27 card, SEAL colours)
    ├── lib/
    │   ├── utils.ts                 # cn() Tailwind merge helper
    │   └── api.ts                   # all fetch calls — BASE_URL = http://localhost:8000
    ├── hooks/
    │   ├── use-mobile.tsx           # (from Lovable)
    │   ├── use-toast.ts             # (from Lovable)
    │   ├── useSimStream.ts          # NEW — WS /ws/simulate/{job_id}
    │   └── useTrainStream.ts        # NEW — WS /ws/train/{job_id}
    ├── components/
    │   ├── DashboardLayout.tsx      # Sidebar + TopNav + content wrapper (from Lovable)
    │   ├── AppSidebar.tsx           # Collapsible sidebar (from Lovable)
    │   ├── TopNav.tsx               # Fixed header (from Lovable)
    │   ├── NavLink.tsx              # Active-state nav link (from Lovable)
    │   ├── SimCanvas.tsx            # NEW — Canvas2D renderer: roads, vehicles, TLS signals
    │   └── ui/                      # 40+ shadcn/ui primitives (from Lovable, untouched)
    └── pages/
        ├── Index.tsx                # / — stat cards, How It Works, CTAs (from Lovable, wire stats)
        ├── Simulation.tsx           # /simulation — replace SVG placeholder with SimCanvas + useSimStream
        ├── Compare.tsx              # /compare — two SimCanvas + two useSimStream + diff panel
        ├── Training.tsx             # /training — wire useTrainStream to chart; POST /api/train
        ├── Communication.tsx        # /communication — charts + table (from Lovable, wire API data)
        └── NotFound.tsx             # (from Lovable)
```

---

## Tech Stack

| Layer | Choice | Reason |
|-------|--------|--------|
| Frontend framework | React 18 + Vite | Fast dev, Canvas API + Recharts coexist cleanly |
| UI scaffold source | LovableOutput (shadcn/ui) | All 5 pages already built with mock data; copy then wire up |
| Component library | shadcn/ui | Pre-built primitives (card, select, table, slider…) already in LovableOutput |
| Simulation rendering | HTML Canvas 2D API | Handles hundreds of moving vehicles without SVG lag |
| Charts | Recharts | Reward curves, comm cost charts — already wired in LovableOutput pages |
| Styling | Tailwind CSS | Dark theme + SEAL accent colours already configured in LovableOutput |
| Backend | FastAPI + uvicorn | Async, native WebSocket, matches Python ecosystem |
| Simulation control | TraCI (via `seal/` copy) | Already used by SEAL; gives vehicle positions, TLS state per step |

---

## API Contract

> Backend owns this contract. Frontend builds to it exactly.

### REST

#### `GET /api/networks`
```json
{ "networks": ["grid-3x3", "grid-5x5", "grid-7x7"] }
```

#### `GET /api/network/{topology}`
Parsed node positions and road edges from the `.net.xml` file. Used to draw the road layout on the canvas before any simulation starts.
```json
{
  "topology": "grid-3x3",
  "bounds": { "min_x": 0, "min_y": 0, "max_x": 600, "max_y": 600 },
  "nodes": [
    { "id": "A0", "x": 100.0, "y": 100.0 }
  ],
  "edges": [
    { "id": "e0", "from": "A0", "to": "A1", "lanes": [
      { "id": "e0_0", "shape": [[100,100],[300,100]] }
    ]}
  ]
}
```

#### `GET /api/weights`
```json
{
  "weights": [
    { "id": "fedrl-naive-3x3-ranked", "trainer": "FedRL", "aggr": "naive",
      "topology": "grid-3x3", "ranked": true },
    { "id": "marl-3x3-ranked", "trainer": "MARL", "aggr": null,
      "topology": "grid-3x3", "ranked": true },
    { "id": "sarl-3x3-ranked", "trainer": "SARL", "aggr": null,
      "topology": "grid-3x3", "ranked": true }
  ]
}
```

#### `POST /api/simulate`
Starts a SUMO simulation episode using the specified weights. The simulation runs headless; state is streamed via WebSocket.
```json
// Request
{ "weight_id": "fedrl-naive-3x3-ranked", "topology": "grid-3x3", "seed": 42 }

// Response
{ "job_id": "sim_abc123", "status": "running" }
```

#### `POST /api/train`
Starts a new training run in the background.
```json
// Request
{ "trainer": "FedRL", "aggr": "naive", "topology": "grid-3x3",
  "ranked": true, "n_episodes": 50, "fed_step": 1 }

// Response
{ "job_id": "train_xyz789", "status": "running" }
```

#### `GET /api/results/{job_id}`
```json
{
  "job_id": "train_xyz789",
  "status": "complete",
  "type": "training",
  "rewards": [
    { "episode": 1, "mean_reward": -3.2, "trainer": "FedRL", "fed_round": true },
    { "episode": 2, "mean_reward": -2.9, "trainer": "FedRL", "fed_round": false }
  ],
  "trip_metrics": {
    "travel_time_s": 99.2,
    "waiting_time_s": 27.1,
    "time_loss_s": 23.4
  },
  "comm_costs": {
    "EDGE2TLS_POLICY": 450,
    "TLS2EDGE_OBS": 360,
    "EDGE2TLS_RANK": 720,
    "EDGE2TLS_ACTION": 360,
    "VEH2TLS": 1200
  }
}
```

### WebSocket

#### `WS /ws/simulate/{job_id}`
Streams one message per simulation timestep. Sent at ~10 fps (every 10 SUMO steps).
```json
{
  "step": 42,
  "done": false,
  "vehicles": [
    { "id": "veh_0", "x": 120.5, "y": 80.2, "speed": 13.5, "angle": 90.0 }
  ],
  "traffic_lights": [
    { "id": "A0", "state": "GGrr", "lane_occupancy": 0.42,
      "halted_lane_occupancy": 0.18, "reward": -0.36, "global_rank": 2 }
  ],
  "metrics": {
    "total_halted": 12,
    "mean_speed": 8.3,
    "mean_reward": -0.24
  }
}
```
When `done: true`, connection closes.

#### `WS /ws/train/{job_id}`
Streams one message per completed training episode.
```json
{ "episode": 12, "mean_reward": -1.24, "trainer": "FedRL", "fed_round": true }
```
Connection closed by server when all episodes complete.

---

## Pages

### 1. Dashboard `/`
The landing page. Explains the project to someone who has never seen it.

- Top nav: "SEAL" logo left, page links right
- Hero section with 3-line explainer: what SUMO is, what the AI does, what FedRL achieves
- 4 stat cards: Communication Reduction 36.24% (teal) / Reward vs MARL −2.11% (amber) / Travel Time Improvement 18.14% (blue) / Active Intersections 49 (purple)
- "How it works" row: 3 cards — "1. Simulate" (SUMO runs a city grid) / "2. Learn" (PPO trains each light) / "3. Federate" (agents share weights periodically)
- Bottom: 2 primary CTAs — "Watch a Simulation →" and "Start Training →"

### 2. Simulation `/simulation`
The centrepiece. Shows a live animated bird's-eye view of a SUMO simulation.

- **Controls bar:** policy dropdown (FedRL-naive / MARL / SARL / Fixed Timing), topology dropdown, seed input, "Run" button, play/pause, speed multiplier (1× 2× 5×)
- **Main canvas (Canvas 2D):**
  - Roads drawn as dark grey rectangles from `GET /api/network/{topology}` edge shapes
  - Intersection circles at node positions, colour = current TLS phase (green/amber/red)
  - Vehicles as small white/yellow rectangles (rotated by angle) moving along roads in real time
  - Congestion heatmap overlay (optional toggle): lanes turn red when lane_occupancy > 0.7
- **Right sidebar (live metrics):**
  - Halted vehicles count (updates every frame)
  - Mean speed (m/s)
  - Cumulative mean reward (line sparkline)
  - Current timestep / total
- **Bottom bar:** scrollable reward-over-time mini chart updating live
- Data from `WS /ws/simulate/{job_id}`

### 3. Compare `/compare`
Run two policies simultaneously on the same seed and watch them diverge.

- **Controls bar:** Policy A dropdown / Policy B dropdown / topology / seed / "Run Both" button
- **Split screen:** two `SimCanvas` side by side, Policy A left, Policy B right — same road layout, independent vehicle + TLS streams
- **Live diff panel** (between the two canvases):
  - Halted: A=14 vs B=9
  - Mean speed: A=7.2 vs B=9.8
  - Reward: A=−0.41 vs B=−0.22
  - "Policy B is performing better right now" badge
- **Bottom:** single dual-line chart — cumulative reward for A (amber) and B (teal) over simulation timesteps
- Runs two separate WebSocket connections simultaneously, same seed ensures identical traffic generation

### 4. Training `/training`
Trigger a new training run and watch the AI improve in real time.

- **Controls:** trainer dropdown (FedRL-naive / FedRL-pos_reward / MARL / SARL), topology, ranked toggle, n_episodes slider (10–100), "Start Training" button
- **Live reward chart:** x = episode number, y = mean reward; line updates as each episode streams in via `WS /ws/train/{job_id}`; dashed vertical lines at federation rounds for FedRL runs
- **Status bar:** current episode / total, current mean reward, estimated time remaining
- **Pre-loaded results toggle:** "Show existing results" — loads stored training curves for all trainers from mock data (FedRL-naive / MARL / SARL) for comparison without running SUMO
- After training completes: "Simulate this policy →" button that takes the new weights to the Simulation page
- Summary table at bottom: Trainer | Final Reward | Best Reward | Episodes | Aggr Fn

### 5. Communication `/communication`
The research metric page — explains why FedRL is worth using.

- **Banner:** "FedRL reduces communication by 36.24% with only 2.11% reward loss vs MARL"
- **Left:** stacked bar chart — x = trainer, y = total message count, stacked by type:
  - EDGE→TLS Policy (blue)
  - TLS→EDGE Observation (teal)
  - EDGE→TLS Rank (green)
  - EDGE→TLS Action (amber)
  - Vehicle→TLS Count (purple)
- **Right:** scatter plot — x = total comm cost, y = mean episode reward; one point per trainer; Pareto frontier curve; FedRL-naive point labelled "★ Best trade-off"
- **Below:** plain-English explanation of what each message type means and why it costs communication bandwidth
- Raw data table: Trainer | EDGE2TLS | TLS2EDGE | RANK | ACTION | VEH2TLS | Total | Reward

---

## Design System

- Background `#0f1117`, card surface `#1a1d27`, border `#2d2f3e`
- FedRL = teal `#14b8a6`, MARL = amber `#f59e0b`, SARL = purple `#a855f7`, primary action = blue `#3b82f6`
- Road colour on canvas: `#1e2030`, vehicle colour: `#e2e8f0`, halted vehicle: `#ef4444`
- TLS colours on canvas: green phase `#22c55e`, yellow `#eab308`, red `#ef4444`
- Monospace for all live metric values; sans-serif for labels
- Desktop-first; canvas is min 600px wide

---

## Agent Ownership

### Backend Agent
- **Owns:** `BackEnd/`
- **Does NOT touch:** `FrontEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Responsibilities:**
  1. Copy `seal/`, `configs/SMARTCOMP/`, `example_weights/ICCPS/Final/`, `netfiles.py`
  2. Implement all REST endpoints and both WebSockets per contract above
  3. Parse `.net.xml` for node positions AND lane shapes (needed for road drawing on canvas)
  4. Wrap SUMO+TraCI into an async generator that yields one frame per N steps
  5. Wrap training into an async background task that yields one result per episode
  6. CORS enabled for `http://localhost:5173`
  7. Job store (`jobs.py`) tracks running and completed jobs by ID

### Frontend Agent
- **Owns:** `FrontEnd/`
- **Does NOT touch:** `BackEnd/`, `SUMO-FedRL-main/`, `LovableOutput/`
- **Note:** `LovableOutput/` was already copied into `FrontEnd/` during initial build. Do not re-copy — only edit `FrontEnd/` files directly.
- **Responsibilities:**
  1. Copy `LovableOutput/seal-traffic-flow-main/` into `FrontEnd/` as the starting scaffold — do not regenerate from scratch
  2. Add `src/lib/api.ts` — typed wrappers for all REST calls (`BASE_URL = http://localhost:8000`)
  3. `SimCanvas.tsx`: Canvas 2D renderer — draw roads from `GET /api/network/{topology}`, animate vehicles each frame, colour TLS circles by phase state
  4. `useSimStream.ts`: WebSocket hook that buffers frames and drives canvas animation via `requestAnimationFrame` — interpolate vehicle positions between frames for smooth 60fps
  5. `useTrainStream.ts`: WebSocket hook that appends episode data to reward chart state
  6. Wire each page to the live API: replace mock data with `fetch` calls, hook `useSimStream`/`useTrainStream` into Simulation, Compare, and Training pages
  7. Leave `src/components/ui/` and layout components (`DashboardLayout`, `AppSidebar`, `TopNav`) untouched unless a bug fix is required

---

## Frontend Scaffold Notes

The UI scaffold already exists in `LovableOutput/seal-traffic-flow-main/`. All 5 pages are built with mock data and the full design system (dark theme, SEAL accent colours, shadcn/ui components, Recharts charts, collapsible sidebar). The frontend agent's job is to **port this into `FrontEnd/` and wire it to the live API**, not to build from scratch.

### What LovableOutput already has (do not rebuild):
- All layout components: `DashboardLayout`, `AppSidebar`, `TopNav`, `NavLink`
- All 5 pages with correct structure, charts, controls, and mock data
- Tailwind dark theme configured with `#0f1117` background, `#1a1d27` card, SEAL accent colours
- 40+ shadcn/ui primitives in `src/components/ui/`
- React Router routes, React Query setup, Sonner toast notifications

### What the frontend agent must add on top:
| File to create | Purpose |
|---------------|---------|
| `src/lib/api.ts` | Typed REST wrappers (`BASE_URL = http://localhost:8000`); replace all mock `// TODO` fetch calls |
| `src/hooks/useSimStream.ts` | WebSocket hook for `/ws/simulate/{job_id}`; drives `SimCanvas` via `requestAnimationFrame` |
| `src/hooks/useTrainStream.ts` | WebSocket hook for `/ws/train/{job_id}`; appends episode data to Training chart |
| `src/components/SimCanvas.tsx` | Canvas 2D renderer — draw roads from network layout, animate vehicles, colour TLS circles |

### Per-page wiring tasks:
- **Simulation.tsx** — replace SVG placeholder with `<SimCanvas>` + connect `useSimStream`; POST `/api/simulate` on Run
- **Compare.tsx** — two `<SimCanvas>` + two `useSimStream` instances with same seed
- **Training.tsx** — POST `/api/train` on Start; connect `useTrainStream` to chart; add "Simulate this policy" handoff
- **Communication.tsx** — replace mock comm data with `GET /api/results/{job_id}` from a completed training run
- **Index.tsx** — stat cards can stay hardcoded (paper values); CTAs already link correctly

---

## Validation

### Backend
```bash
cd BackEnd
uvicorn api.main:app --reload
curl http://localhost:8000/api/networks
curl http://localhost:8000/api/network/grid-3x3   # must include lane shapes
curl http://localhost:8000/api/weights
curl -X POST http://localhost:8000/api/simulate \
  -H "Content-Type: application/json" \
  -d '{"weight_id":"fedrl-naive-3x3-ranked","topology":"grid-3x3","seed":42}'
# → { "job_id": "...", "status": "running" }
# Then connect WebSocket to ws://localhost:8000/ws/simulate/{job_id}
# → confirm vehicle + TLS JSON frames arrive
```

### Frontend
```bash
cd FrontEnd
npm run build    # 0 errors
npm run dev      # starts on :5173
# Dashboard: both CTA buttons navigate correctly
# Simulation: canvas placeholder visible, controls bar renders
# Compare: split-screen layout renders, diff panel visible
# Training: chart renders with mock data, trainer dropdown works
# Communication: both charts render, Pareto point visible
```

### End-to-End
1. Backend on `:8000`, frontend on `:5173`
2. Simulation page: select FedRL-naive + grid-3x3 + seed 42 → Run → vehicles appear on canvas and move → sidebar metrics update live
3. Compare page: Policy A = MARL, Policy B = FedRL-naive, same seed → both canvases animate → diff panel shows which is better
4. Training page: select FedRL-naive → Start Training → chart line extends episode by episode in real time
5. After training: "Simulate this policy" navigates to Simulation with new weights loaded

---

## Acceptance Criteria

- [ ] All 5 pages render without console errors
- [ ] SimCanvas animates vehicles in real time from WebSocket frames
- [ ] Compare page runs two WebSocket connections simultaneously with same seed
- [ ] Training chart extends live as episodes complete
- [ ] `/api/network/{topology}` returns lane shapes sufficient to draw road layout
- [ ] `/ws/simulate/{job_id}` delivers vehicle positions at ≥5 fps equivalent
- [ ] `SUMO-FedRL-main/` has zero modifications
- [ ] `LovableOutput/` has zero modifications
- [ ] Backend only imports from its own copied `seal/` directory

---

## Notes

- SUMO must be installed on the host machine (not pip-installable). `sumo` binary must be on PATH. Backend README must document this with install instructions for Windows/Linux/Mac.
- For the Compare page, both simulations use the same random seed — this is achievable because `AbstractSumoEnv` accepts a seed parameter for route generation.
- The canvas animation runs at the browser's `requestAnimationFrame` rate (~60fps) but only has new data every ~10 SUMO steps. The frontend should interpolate vehicle positions between frames for smooth animation.
- `fed_step=1` in the code = aggregation every episode (360 steps). The paper's "4000 timesteps" is a different configuration. Both are valid; `fed_step` is a configurable parameter.
- The Fixed Timing baseline in Compare is just SUMO running with its default timed phases and no RL policy applied.
